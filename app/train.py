import os
import sys
import torch
from datetime import datetime
import yaml
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler



project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.datasets import VideoDataset
from models.pico_jepa import PicoJEPA_Pretrain
from utils.utils import create_dummy_dataset_if_needed, print_system_info


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Load Configuration ---
config_path = os.path.join(project_root, "configs", "config.yaml")
CONFIG = load_config(config_path)


def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, "a") as f:
        f.write(full_message + "\n")


def save_model(pico_jepa_model, encoder_save_path):
    print(f"Saving pre-trained online_encoder weights to {encoder_save_path}...")
    torch.save(pico_jepa_model.online_encoder.state_dict(), encoder_save_path)
    print(f"Pre-trained encoder saved successfully to {encoder_save_path}")


def do_pretraining():
    log_file_name = "training_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".log"
    device = print_system_info(force_cpu=CONFIG["force_cpu"])
    print("Initializing Video Dataset for Pre-training (with torchcodec 0.4.0 API)...")
    video_dataset = VideoDataset(
        video_dir=CONFIG["video_dir"],
        csv_file=CONFIG["csv_file"],
        frames_per_clip=CONFIG["frames_per_clip"],
        target_height=CONFIG["resize_height"],
        target_width=CONFIG["resize_width"],
        channels=CONFIG["video_channels"],
        labeled=False,
        sampling_strategy="random",
    )

    if len(video_dataset) == 0:
        print(
            f"Error: Dataset is empty. Please check '{CONFIG['video_dir']}' and '{CONFIG['csv_file']}'."
        )
        csv_path = os.path.join(CONFIG["video_dir"], CONFIG["csv_file"])
        if not os.path.exists(CONFIG["video_dir"]) or not os.path.exists(csv_path):
            print(
                "Hint: The dataset directory or CSV file might be missing. The script created a dummy CSV if it was absent."
            )
        exit()

    data_loader = DataLoader(
        video_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )
    print(
        f"Dataset initialized with {len(video_dataset)} videos. DataLoader ready with {len(data_loader)} batches."
    )

    encoder_save_path = CONFIG["encoder_save_path"]

    pico_jepa_model = PicoJEPA_Pretrain(CONFIG).to(device)
    params_count = sum(
        p.numel() for p in pico_jepa_model.parameters() if p.requires_grad
    )
    print(f"PicoJEPA_Pretrain model created. Trainable parameters: {params_count:,}")

    # --- Learning Rate Optimization: Parameter Groups and Scheduler ---
    base_lr = CONFIG["learning_rate"]
    predictor_lr_multiplier = CONFIG.get("predictor_lr_multiplier", 1.0)
    param_groups = [
        # Group 1: Encoder weights (no bias, no 1D)
        {
            "params": (
                p
                for n, p in pico_jepa_model.online_encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            ),
            "lr": base_lr,
            "weight_decay": CONFIG["weight_decay"],
        },
        # Group 2: Predictor weights (no bias, no 1D)
        {
            "params": (
                p
                for n, p in pico_jepa_model.predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            ),
            "lr": base_lr
            * predictor_lr_multiplier,  # Potentially higher LR for the predictor
            "weight_decay": CONFIG["weight_decay"],
        },
        # Group 3: Bias and 1D Encoder parameters (weight_decay = 0)
        {
            "params": (
                p
                for n, p in pico_jepa_model.online_encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "lr": base_lr,  # The bias can follow the scheduler or have a fixed LR
            "weight_decay": 0,
        },
        # Group 4: Bias and 1D Predictor parameters (weight_decay = 0)
        {
            "params": (
                p
                for n, p in pico_jepa_model.predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "lr": base_lr
            * predictor_lr_multiplier,  # The bias can follow the scheduler or have a fixed LR
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,

    )

    total_steps = len(data_loader) * CONFIG["num_epochs"]
    warmup_epochs = CONFIG.get("warmup_epochs", int(0.1 * CONFIG["num_epochs"]))
    warmup_steps = warmup_epochs * len(data_loader)

    # final_lr, if it does not exist by default it will be 0.
    final_lr = CONFIG.get("final_lr", 0.0)
    # T_max is the number of steps for the scheduler after the warmup.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=final_lr,
    )
    print(
        f"\n--- Starting PicoJEPA_ViT Self-Supervised Pre-training for {CONFIG['num_epochs']} epochs ---"
    )
    log_message(
        f"Dataset initialized with {len(video_dataset)} videos. DataLoader ready with {len(data_loader)} batches.",
        log_file_name,
    )
    log_message(
        f"Optimizer: AdamW | Base LR: {base_lr} | Predictor LR Multiplier: {predictor_lr_multiplier} | Weight Decay: {CONFIG['weight_decay']}",
        log_file_name,
    )
    log_message(
        f"Scheduler: CosineAnnealingLR with {warmup_epochs} warmup epochs. Final LR: {final_lr}",
        log_file_name,
    )
    if CONFIG.get("clip_grad_norm") is not None:
        log_message(
            f"Gradient Clipping: Enabled with norm {CONFIG['clip_grad_norm']} (after warmup epochs).",
            log_file_name,
        )
    else:
        log_message(
            "Gradient Clipping: Disabled.",
            log_file_name,
        )

    pico_jepa_model.train()

    for epoch in range(CONFIG["num_epochs"]):
        total_epoch_loss = 0
        batches_processed = 0
        for i, videos_batch in enumerate(data_loader):
            if videos_batch is None or (
                isinstance(videos_batch, torch.Tensor) and videos_batch.nelement() == 0
            ):
                print(f"Warning: Received an empty or None batch {i + 1}. Skipping.")
                continue
            current_step = epoch * len(data_loader) + i
            # ---  Learning Rate Management (Warmup and Scheduler)---
            if current_step < warmup_steps:
                # Linear Warmup Phase: Increase LR from 0 to base_lr
                warmup_factor = current_step / warmup_steps
                for param_group in optimizer.param_groups:
                    # Apply the warmup factor to the base LR of each group
                    # Make sure param_group['lr'] is set to base
                    param_group["lr"] = param_group["initial_lr"] * warmup_factor
            else:
                # After the warmup, the scheduler takes over
                scheduler.step()

            if (i + 1) % 50 == 0:  #
                current_lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
                print(f"Current LRs: {', '.join(current_lrs)}")

            videos_batch = videos_batch.to(device, non_blocking=True)
            expected_shape = (
                CONFIG["batch_size"],
                CONFIG["video_channels"],
                CONFIG["frames_per_clip"],
                CONFIG["resize_height"],
                CONFIG["resize_width"],
            )
            if videos_batch.shape != expected_shape:
                print(
                    f"Warning: Batch {i + 1} has unexpected shape {videos_batch.shape}. Expected {expected_shape}. Skipping."
                )
                continue

            optimizer.zero_grad()
            loss = pico_jepa_model(videos_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: NaN or Inf loss encountered at Epoch {epoch + 1}, Batch {i + 1}. Skipping update."
                )
                continue

            loss.backward()
            clip_grad_norm_value = CONFIG.get("clip_grad_norm")
            if (
                clip_grad_norm_value is not None and epoch >= warmup_epochs
            ):  # Apply clipping after the warmup
                # Applies to all model parameters to simplify
                # If mixed precision (scaler) were used, scaler.unscale_(optimizer) would go here before the clip
                torch.nn.utils.clip_grad_norm_(
                    pico_jepa_model.parameters(), clip_grad_norm_value
                )


            optimizer.step()

            total_epoch_loss += loss.item()
            batches_processed += 1

            if (i + 1) % 5 == 0 or i == len(data_loader) - 1:
                print(
                    f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | Batch {i + 1}/{len(data_loader)} | Loss: {loss.item():.4f}"
                )
                log_message(
                    f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | Batch {i + 1}/{len(data_loader)} | Loss: {loss.item():.4f}",
                    log_file_name,
                )

        if batches_processed > 0:
            avg_epoch_loss = total_epoch_loss / batches_processed
            print(
                f"--- Epoch {epoch + 1} Summary --- Avg. Loss: {avg_epoch_loss:.4f} ---"
            )
            log_message(
                f"Epoch {epoch + 1} Summary: Avg. Loss: {avg_epoch_loss:.4f}",
                log_file_name,
            )
        else:
            print(
                f"--- Epoch {epoch + 1} Summary --- No batches were processed. Check data loading and dataset contents. ---"
            )
            log_message(
                f"Epoch {epoch + 1} Summary: No batches were processed. Check data loading and dataset contents.",
                log_file_name,
            )

        # Save the pre-trained online_encoder
        save_model(pico_jepa_model, encoder_save_path)

    print("\n--- PicoJEPA_ViT Self-Supervised Pre-training finished. ---")

    # Save the pre-trained online_encoder
    save_model(pico_jepa_model, encoder_save_path)

    print("\nImportant Notes:")
    print(
        "1. This script performs self-supervised pre-training using the PicoJEPA method."
    )
    print(
        "2. Ensure 'torchcodec' (v0.4.0 in this case) and its dependencies (like a compatible FFmpeg) are correctly installed."
    )
    print(
        "3. For actual training, ensure your dataset path and CSV (e.g., nano-train.csv with video names) are correct."
    )
    print(
        "4. Monitor memory usage. If OOM errors occur, reduce batch_size, video dimensions, or ViT model size in CONFIG."
    )
    print(
        f"5. The pre-trained encoder is saved to '{encoder_save_path}' and can be used for downstream tasks like classification."
    )


if __name__ == "__main__":
    do_pretraining()
