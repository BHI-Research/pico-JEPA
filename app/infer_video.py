import argparse
import os
import sys
import torch
import yaml


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.utils import print_system_info
from models.video_classifier import VideoClassifier
from dataset.datasets import load_and_preprocess_single_video


def load_config_from_yaml(config_path):

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file was not found in: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class_names_from_file(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The class name file was not found at: {file_path}")

    class_names = []
    with open(file_path, "r") as f:
        for line in f:

            clean_line = line.strip()
            if not clean_line:
                continue

            parts = clean_line.split()
            if len(parts) > 1 and parts[0].isdigit():
                class_names.append(" ".join(parts[1:]))
            else:
                class_names.append(clean_line)
    return class_names


def do_inference(args):

    try:
        CONFIG_INFER = load_config_from_yaml(args.config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return


    CONFIG_INFER["force_cpu"] = args.force_cpu if args.force_cpu is not None else CONFIG_INFER.get("force_cpu", False)
    

    try:
        class_names_list = load_class_names_from_file(args.class_names_file)
        CONFIG_INFER["num_classes"] = len(class_names_list)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    device = print_system_info(CONFIG_INFER)
    print(f"Using the device: {device}")

    model = VideoClassifier(
        encoder_config=CONFIG_INFER,
        num_classes=CONFIG_INFER["num_classes"],
        pretrained_encoder_path=None,
    ).to(device)

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: The model file was not found in {args.model_path}")
        exit()
    except Exception as e:
        print(f"Error loading model state: {e}")
        exit()

    model.eval()


    video_tensor = load_and_preprocess_single_video(
        args.video_path,CONFIG_INFER
    )
    
    video_tensor = video_tensor.to(device).unsqueeze(0)
    print("Performing inference...")
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    predicted_class_idx = predicted_class_idx.item()
    confidence = confidence.item()
    print("\n--- Inference Result ---")
    print(f"Video: {os.path.basename(args.video_path)}")

    if 0 <= predicted_class_idx < len(class_names_list):
          predicted_class_name = class_names_list[predicted_class_idx]
          print(
                f"Predicted Class: {predicted_class_name} (Index: {predicted_class_idx})"
          )
    else:
          print(
                f"Predicted Class Index: {predicted_class_idx} (Class name mapping not found or out of range)"
            )
    print(f"Confidence: {confidence:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PicoJEPA video classification inference."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Route to the trained classifier model (.pth file).",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file YAML (e.g., configs/config.yaml).",
    )
    # --- Nuevo argumento para la ruta del archivo de nombres de clase ---
    parser.add_argument(
        "--class_names_file",
        type=str,
        required=True,
        help="Path to a text or CSV file with the list of class names.",
    )
    # -------------------------------------------------------------------
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if a GPU is available."
    )
    
    args = parser.parse_args()
    do_inference(args)

