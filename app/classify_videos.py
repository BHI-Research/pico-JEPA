import argparse
import os
import sys
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim.lr_scheduler


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.datasets import VideoDataset
from models.video_classifier import VideoClassifier
from utils.utils import create_dummy_dataset_if_needed, print_system_info


def load_config_from_yaml(config_path):

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file was not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, "a") as f:
        f.write(full_message + "\n")


# --- Main Classification Training Script ---
def train_video_classifier(args):
    try:
        CONFIG_CLASSIFY = load_config_from_yaml(args.config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    device = print_system_info(force_cpu=CONFIG_CLASSIFY["force_cpu"])
    log_file_name = (
        "classification" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".log"
    )
    classifier_save_path = CONFIG_CLASSIFY.get(
        "classifier_save_path", "./pico_jepa_classifier.pth"
    )

    print("Initializing Labeled Video Dataset for Classification...")
    classify_dataset = VideoDataset(
        video_dir=CONFIG_CLASSIFY["video_dir"],
        csv_file=CONFIG_CLASSIFY["csv_file_labeled"],
        frames_per_clip=CONFIG_CLASSIFY["frames_per_clip"],
        target_height=CONFIG_CLASSIFY["resize_height"],
        target_width=CONFIG_CLASSIFY["resize_width"],
        channels=CONFIG_CLASSIFY["video_channels"],
        labeled=True,
        sampling_strategy="random",
    )

    if len(classify_dataset) == 0:
        print(
            f"Error: Classification dataset is empty. Please check '{CONFIG_CLASSIFY['video_dir']}' and '{CONFIG_CLASSIFY['csv_file_labeled']}'."
        )
        exit()

    # Simple 80/20 train/validation split
    train_size = int(0.8 * len(classify_dataset))
    val_size = len(classify_dataset) - train_size

    if train_size == 0 or val_size == 0:
        print(
            f"Dataset too small for train/val split (Total: {len(classify_dataset)}). Needs at least 2 samples for split."
        )
        print("Proceeding with the full dataset for training, no validation.")
        train_dataset = classify_dataset
        val_dataset = None  # No validation set
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            classify_dataset, [train_size, val_size]
        )

    print(f"Training set size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG_CLASSIFY["batch_size"],
        shuffle=True,
        num_workers=CONFIG_CLASSIFY["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
        drop_last=(
            True if len(train_dataset) > CONFIG_CLASSIFY["batch_size"] else False
        ),  # Avoid error if dataset smaller than batch
    )
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG_CLASSIFY["batch_size"],
            shuffle=False,
            num_workers=CONFIG_CLASSIFY["num_workers"],
            pin_memory=True if device.type == "cuda" else False,
            drop_last=False,
        )
    else:
        val_loader = None

    print("DataLoader for classification ready.")

    classification_model = VideoClassifier(
        encoder_config=CONFIG_CLASSIFY,  # Pass the classification config
        num_classes=CONFIG_CLASSIFY["num_classes"],
        freeze_encoder=CONFIG_CLASSIFY[
            "freeze_encoder"
        ],  # This will be handled by the class
        pretrained_encoder_path=CONFIG_CLASSIFY["encoder_save_path"],
    ).to(device)

    # --- Optimizer Configuration and Parameter Groups ---
    params_to_optimize = []
    if not CONFIG_CLASSIFY["freeze_encoder"]:
        print("Setting up optimizer for fine-tuning encoder and classification head.")

        params_to_optimize.append(
            {
                "params": (
                    p
                    for n, p in classification_model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ),
                "lr": CONFIG_CLASSIFY["learning_rate_encoder_finetune"],
                "weight_decay": CONFIG_CLASSIFY["classify_weight_decay"],
            }
        )
        params_to_optimize.append(
            {
                "params": (
                    p
                    for n, p in classification_model.encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "lr": CONFIG_CLASSIFY["learning_rate_encoder_finetune"],
                "weight_decay": 0,
            }
        )
    else:
        print("Encoder is frozen. Setting up optimizer for classification head only.")
        pass

    params_to_optimize.append(
        {
            "params": (
                p
                for n, p in classification_model.classifier_head.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            ),
            "lr": CONFIG_CLASSIFY["learning_rate_classifier"],
            "weight_decay": CONFIG_CLASSIFY["classify_weight_decay"],
        }
    )
    params_to_optimize.append(
        {
            "params": (
                p
                for n, p in classification_model.classifier_head.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "lr": CONFIG_CLASSIFY["learning_rate_classifier"],
            "weight_decay": 0,
        }
    )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss()

    # ---  Configuring the Learning Rate Scheduler with Warmup ---
    total_steps_classify = len(train_loader) * CONFIG_CLASSIFY["num_epochs_classify"]
    warmup_epochs_classify = CONFIG_CLASSIFY.get("warmup_epochs_classify", 0)
    warmup_steps_classify = warmup_epochs_classify * len(train_loader)
    final_lr_classify = CONFIG_CLASSIFY.get("final_lr_classify", 0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps_classify - warmup_steps_classify,
        eta_min=final_lr_classify,
    )

    print(
        f"\n--- Starting Video Classification Training for {CONFIG_CLASSIFY['num_epochs_classify']} epochs ---"
    )
    log_message(
        f"Training set size: {len(train_dataset)}. DataLoader ready with {len(train_loader)} batches.",
        log_file_name,
    )
    log_message(
        f"Optimizer: AdamW | Encoder Fine-tune LR: {CONFIG_CLASSIFY['learning_rate_encoder_finetune']} | Classifier Head LR: {CONFIG_CLASSIFY['learning_rate_classifier']} | Weight Decay: {CONFIG_CLASSIFY['classify_weight_decay']}",
        log_file_name,
    )
    log_message(
        f"Scheduler: CosineAnnealingLR with {warmup_epochs_classify} warmup epochs. Final LR: {final_lr_classify}",
        log_file_name,
    )
    if CONFIG_CLASSIFY.get("clip_grad_norm_classify") is not None:
        log_message(
            f"Gradient Clipping: Enabled with norm {CONFIG_CLASSIFY['clip_grad_norm_classify']} (after warmup epochs).",
            log_file_name,
        )
    else:
        log_message(
            "Gradient Clipping: Disabled.",
            log_file_name,
        )

    for epoch in range(CONFIG_CLASSIFY["num_epochs_classify"]):
        classification_model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0

        if not train_loader:
            print(
                "Error: Train loader is not initialized (dataset might be too small)."
            )
            break

        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            videos_batch, labels_batch = batch_data
            if videos_batch is None or videos_batch.nelement() == 0:
                continue

            current_step_classify = epoch * len(train_loader) + i

            # ---  Learning Rate Management (Warmup and Scheduler) ---
            if current_step_classify < warmup_steps_classify:
                warmup_factor = current_step_classify / warmup_steps_classify
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["initial_lr"] * warmup_factor
            else:
                scheduler.step()

            if (i + 1) % 50 == 0:
                current_lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
                print(f"Current LRs: {', '.join(current_lrs)}")

            videos_batch = videos_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = classification_model(videos_batch)
            loss = criterion(logits, labels_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: NaN or Inf loss in training (Epoch {epoch + 1}, Batch {i + 1}). Skipping batch."
                )
                continue

            loss.backward()

            clip_grad_norm_value = CONFIG_CLASSIFY.get("clip_grad_norm_classify")

            if clip_grad_norm_value is not None and epoch >= warmup_epochs_classify:
                torch.nn.utils.clip_grad_norm_(
                    classification_model.parameters(), clip_grad_norm_value
                )

            optimizer.step()

            total_train_loss += loss.item()
            _, predicted_labels = torch.max(logits, 1)
            correct_train_predictions += (predicted_labels == labels_batch).sum().item()
            total_train_samples += labels_batch.size(0)

            if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
                print(
                    f"Epoch {epoch + 1} [Train] | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )
                log_message(
                    f"Epoch {epoch + 1} [Train] | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}",
                    log_file_name,
                )

        avg_train_loss = (
            total_train_loss / len(train_loader)
            if len(train_loader) > 0
            else float("inf")
        )
        train_accuracy = (
            (correct_train_predictions / total_train_samples) * 100
            if total_train_samples > 0
            else 0.0
        )
        print(
            f"--- Epoch {epoch + 1} [Train] Summary --- Avg. Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2f}% ---"
        )
        log_message(
            f"--- Epoch {epoch + 1} [Train] Summary --- Avg. Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2f}% ---",
            log_file_name,
        )

        if val_loader:
            classification_model.eval()
            total_val_loss = 0
            correct_val_predictions = 0
            total_val_samples = 0
            with torch.no_grad():
                for videos_batch, labels_batch in val_loader:
                    if videos_batch is None or videos_batch.nelement() == 0:
                        continue
                    videos_batch = videos_batch.to(device, non_blocking=True)
                    labels_batch = labels_batch.to(device, non_blocking=True)

                    logits = classification_model(videos_batch)
                    loss = criterion(logits, labels_batch)
                    total_val_loss += loss.item()
                    _, predicted_labels = torch.max(logits, 1)
                    correct_val_predictions += (
                        (predicted_labels == labels_batch).sum().item()
                    )
                    total_val_samples += labels_batch.size(0)

            avg_val_loss = (
                total_val_loss / len(val_loader)
                if len(val_loader) > 0
                else float("inf")
            )
            val_accuracy = (
                (correct_val_predictions / total_val_samples) * 100
                if total_val_samples > 0
                else 0.0
            )
            print(
                f"--- Epoch {epoch + 1} [Validation] Summary --- Avg. Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}% ---"
            )
            log_message(
                f"--- Epoch {epoch + 1} [Validation] Summary --- Avg. Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}% ---",
                log_file_name,
            )
        else:
            print(
                f"--- Epoch {epoch + 1} [Validation] --- No validation set to evaluate."
            )
        torch.save(classification_model.state_dict(), classifier_save_path)

    print("\n--- Video Classification Training finished. ---")

    print(f"Saving classification model state_dict to: {classifier_save_path}")
    torch.save(classification_model.state_dict(), classifier_save_path)
    print(f"Classification model saved successfully to {classifier_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PicoJEPA video classification."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file YAML (e.g., configs/config.yaml).",
    )
    args = parser.parse_args()
    train_video_classifier(args)
