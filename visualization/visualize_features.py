import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from dataset.datasets import VideoDataset
from models.video_classifier import VideoClassifier
from utils.utils import print_system_info


def load_config_from_yaml(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Load Configuration from YAML ---
config_file_path = os.path.join(project_root, "configs", "config.yaml")
CONFIG_VIZ = load_config_from_yaml(config_file_path)


def visualize_features(args):
    CONFIG_VIZ["num_classes"] = args.num_classes

    device = print_system_info(force_cpu=CONFIG_VIZ.get("force_cpu", True))
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading trained classifier from: {args.classifier_path}")
    classification_model = VideoClassifier(
        encoder_config=CONFIG_VIZ,
        num_classes=CONFIG_VIZ["num_classes"],
        pretrained_encoder_path=None,
    ).to(device)
    try:
        classification_model.load_state_dict(
            torch.load(args.classifier_path, map_location=device)
        )
    except Exception as e:
        print(f"Error loading trained classifier weights: {e}.")
        exit()
    classification_model.eval()

    # 2. Prepare Dataset and Sample
    print("Initializing LABELED dataset...")
    feature_dataset = VideoDataset(
        video_dir=CONFIG_VIZ["video_dir"],
        csv_file=CONFIG_VIZ["csv_file_labeled"],
        frames_per_clip=CONFIG_VIZ["frames_per_clip"],
        target_height=CONFIG_VIZ["resize_height"],
        target_width=CONFIG_VIZ["resize_width"],
        channels=CONFIG_VIZ["video_channels"],
        labeled=True,
        sampling_strategy="center",
    )

    if len(feature_dataset) == 0:
        print("Error: Dataset is empty.")
        exit()

    df = feature_dataset.video_files_df
    df['label'] = df['label'].astype(int)
    
    unique_labels_in_file = sorted(df['label'].unique())
    num_classes_in_file = len(unique_labels_in_file)
    videos_per_class = max(1, args.num_videos // num_classes_in_file if num_classes_in_file > 0 else 1)
    
    all_indices = []
    for class_id in unique_labels_in_file:
        class_indices = df[df['label'] == class_id].index.tolist()
        num_to_sample = min(len(class_indices), videos_per_class)
        if num_to_sample > 0:
            all_indices.extend(random.sample(class_indices, num_to_sample))
            
    random.shuffle(all_indices)
    feature_dataset_subset = Subset(feature_dataset, all_indices)
    print(f"Attempting to process {len(feature_dataset_subset)} videos from {num_classes_in_file} classes...")

    feature_loader = DataLoader(
        feature_dataset_subset, batch_size=1, shuffle=False, num_workers=CONFIG_VIZ["num_workers"]
    )

    # 3. Extract Features (now robust to loading errors)
    all_patch_embeddings = []
    all_ground_truth_clip_labels = []
    failed_videos_count = 0

    with torch.no_grad():
        for data in feature_loader:
            if data is None: 
                failed_videos_count += 1
                continue
            
            video_clip, label = data
            
            # --- MODIFICATION: Check for dummy data from the dataset ---
            # The VideoDataset returns label -1 on failure
            if label.item() == -1:
                failed_videos_count += 1
                continue # Skip this failed video

            video_clip = video_clip.to(device)
            _, patch_embeddings = classification_model(video_clip, return_features=True)
            all_patch_embeddings.append(patch_embeddings.squeeze(0).cpu())
            all_ground_truth_clip_labels.extend([label.item()] * patch_embeddings.shape[1])

    print(f"\nSuccessfully loaded {len(all_patch_embeddings)} videos.")
    if failed_videos_count > 0:
        print(f"Warning: Failed to load {failed_videos_count} videos. Check console for 'Error for video...' messages.")

    if not all_patch_embeddings:
        print("No embeddings were extracted successfully. Cannot visualize.")
        exit()

    concatenated_labels = np.array(all_ground_truth_clip_labels)
    unique_labels, counts = np.unique(concatenated_labels, return_counts=True)
    print("\n--- Feature Extraction Summary ---")
    print(f"Final labels and patch counts in plot: {dict(zip(unique_labels, counts))}")
    print("--------------------------------\n")
    
    concatenated_embeddings = torch.cat(all_patch_embeddings, dim=0).numpy()

    # 4. & 5. Dimensionality Reduction and Plotting (No changes needed)
    tsne = TSNE(n_components=2, perplexity=args.tsne_perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(concatenated_embeddings)

    plt.figure(figsize=(14, 10))
    class_names = [name.strip() for name in args.class_names.split(',')] if args.class_names else None
    cmap = plt.cm.get_cmap("viridis", args.num_classes)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=concatenated_labels, s=10, alpha=0.7, cmap=cmap)
    
    plt.title("t-SNE of Video Patch Embeddings (Colored by Ground Truth Class)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    legend_labels = class_names or [f"Class {i}" for i in range(args.num_classes)]
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i / (args.num_classes - 1 if args.num_classes > 1 else 1)), markersize=8) for i in range(args.num_classes)]
    plt.legend(handles, legend_labels, title="Ground Truth Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = "./patch_embeddings_tsne_colored.png"
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PicoJEPA Feature Visualization.")
    parser.add_argument("--classifier_path", type=str, default=CONFIG_VIZ["classifier_save_path"])
    parser.add_argument("--num_videos", type=int, default=CONFIG_VIZ["num_videos_for_viz"])
    parser.add_argument("--max_patches", type=int, default=CONFIG_VIZ["max_patches_for_viz"])
    parser.add_argument("--tsne_perplexity", type=float, default=CONFIG_VIZ["tsne_perplexity"])
    parser.add_argument("--class_names", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=CONFIG_VIZ["num_classes"])
    args = parser.parse_args()
    visualize_features(args)
