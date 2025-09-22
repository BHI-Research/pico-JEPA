import argparse
import os, sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Add project root to sys.path to resolve module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORT THE NEW CENTRALIZED FUNCTION ---
from dataset.datasets import load_and_preprocess_single_video
from utils.utils import generate_spatiotemporal_masks


def load_config_from_yaml(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Load Configuration from YAML ---
config_file_path = os.path.join(project_root, "configs", "config.yaml")
CONFIG_VIS = load_config_from_yaml(config_file_path)


# --- Mask Generation and Patch Drawing (Logic remains the same) ---
def get_patch_info(config):
    num_patches_t = config["frames_per_clip"] // config["vit_patch_size_t"]
    num_patches_h = config["resize_height"] // config["vit_patch_size_h"]
    num_patches_w = config["resize_width"] // config["vit_patch_size_w"]
    total_spatiotemporal_patches = num_patches_t * num_patches_h * num_patches_w
    return num_patches_t, num_patches_h, num_patches_w, total_spatiotemporal_patches


def draw_occlusion_patches_on_frame(
    frame_chw_np, config, boolean_mask_3d_for_frame, frame_idx_in_clip
):
    """Draws patch grid on a single frame, coloring occluded/visible patches."""
    patch_h_pixels = config["vit_patch_size_h"]
    patch_w_pixels = config["vit_patch_size_w"]

    # frame_chw_np is already [0,1] float, so multiply by 255
    frame_hwc_np = (frame_chw_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_hwc_np, cv2.COLOR_RGB2BGR)

    overlay = frame_bgr.copy()
    output = frame_bgr.copy()

    visible_color = (0, 255, 0)
    occluded_color_border = (0, 0, 255)
    occlusion_fill_alpha = 0.5

    num_patches_h_grid = config["resize_height"] // patch_h_pixels
    num_patches_w_grid = config["resize_width"] // patch_w_pixels

    for r_idx in range(num_patches_h_grid):
        for c_idx in range(num_patches_w_grid):
            is_occluded = boolean_mask_3d_for_frame[r_idx, c_idx].item()
            y1, x1 = r_idx * patch_h_pixels, c_idx * patch_w_pixels
            y2, x2 = y1 + patch_h_pixels, x1 + patch_w_pixels
            if is_occluded:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 128), -1)
                cv2.rectangle(output, (x1, y1), (x2, y2), occluded_color_border, 1)
            else:
                cv2.rectangle(output, (x1, y1), (x2, y2), visible_color, 1)

    cv2.addWeighted(
        overlay, occlusion_fill_alpha, output, 1 - occlusion_fill_alpha, 0, output
    )
    return output


def visulize_patches(args):
    CONFIG_VIS["mask_ratio"] = args.mask_ratio

    device = torch.device("cpu")
    print(f"Using device: {device} for tensor operations.")

    # --- CALL THE NEW CENTRALIZED FUNCTION ---
    # We pass for_visualization=True to skip normalization
    clip_for_viz_cthw = load_and_preprocess_single_video(
        args.video_path, CONFIG_VIS, for_visualization=True
    )

    if clip_for_viz_cthw is None:
        print("Failed to load or preprocess video.")
        exit()

    # Transpose from CTHW to TCHW for visualization loop
    clip_for_viz_tchw = clip_for_viz_cthw.permute(1, 0, 2, 3)
    print(
        f"Video clip loaded. Shape for visualization (TCHW): {clip_for_viz_tchw.shape}"
    )

    # --- Generate Spatio-temporal Mask ---
    num_p_t, num_p_h, num_p_w, total_st_patches = get_patch_info(CONFIG_VIS)
    
    # Generate the tubelet mask using the updated function
    boolean_mask_1d_st = generate_spatiotemporal_masks(
        num_p_t, num_p_h, num_p_w, CONFIG_VIS["mask_ratio"], device, batch_size=1
    )
    boolean_mask_3d_st = boolean_mask_1d_st.reshape(num_p_t, num_p_h, num_p_w)

    # --- Visualization loop (remains the same) ---
    num_frames_to_show = clip_for_viz_tchw.shape[0]
    patch_size_t_pixels = CONFIG_VIS["vit_patch_size_t"]
    cols = min(4, num_frames_to_show)
    rows = (num_frames_to_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.2))
    if num_frames_to_show == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_frames_to_show):
        frame_chw_tensor = clip_for_viz_tchw[i]
        frame_chw_np = frame_chw_tensor.cpu().numpy()
        current_temporal_patch_idx = i // patch_size_t_pixels
        if current_temporal_patch_idx < num_p_t:
            spatial_mask_for_this_temporal_slice = boolean_mask_3d_st[
                current_temporal_patch_idx, :, :
            ]
        else:
            print(
                f"Warning: Frame index {i} is outside the range of defined temporal patches."
            )
            spatial_mask_for_this_temporal_slice = torch.zeros(
                (num_p_h, num_p_w), dtype=torch.bool, device=device
            )

        frame_with_occlusion_bgr = draw_occlusion_patches_on_frame(
            frame_chw_np, CONFIG_VIS, spatial_mask_for_this_temporal_slice, i
        )
        frame_with_occlusion_rgb = cv2.cvtColor(
            frame_with_occlusion_bgr, cv2.COLOR_BGR2RGB
        )

        ax = axes[i]
        ax.imshow(frame_with_occlusion_rgb)
        ax.set_title(f"Frame {i + 1} (Temp. Patch Idx {current_temporal_patch_idx})")
        ax.axis("off")

    for j in range(num_frames_to_show, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    title_text = (
        f"Occlusion Visualization (Tubelet Masking): {os.path.basename(args.video_path)}\n"
        f"Green Border = Visible (Context), Red Border/Overlay = Occluded (Target)\n"
        f"Mask Ratio: {CONFIG_VIS['mask_ratio']:.2f}, Spatio-Temporal Patches: {total_st_patches}"
    )
    plt.suptitle(title_text, fontsize=12, y=0.99)
    fig.subplots_adjust(top=0.90 if rows > 1 else 0.85)
    save_path = "./occlusion visualization.png"
    plt.savefig(save_path)
    print(f"occlusion plot saved to {save_path}")
    plt.show()

    print(f"\nDisplayed {num_frames_to_show} frames.")
    print("Green bordered patches are 'visible' (context).")
    print(
        "Red bordered/overlayed patches are 'occluded' (targets to be predicted by JEPA)."
    )
    print(
        f"The mask is consistent across the temporal dimension (tubelets)."
    )


# --- Main Visualization Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PicoJEPA Video Patch Occlusion Visualization."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=CONFIG_VIS["mask_ratio"],
        help="Proportion of patches to mask.",
    )

    args = parser.parse_args()
    visulize_patches(args)