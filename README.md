# pico-JEPA: Self-Supervised Video Representation Learning and Classification

This project implements a minimal version of a Joint-Embedding Predictive Architecture (JEPA) for self-supervised learning from videos, followed by a classification stage to evaluate the learned representations. It uses `torchcodec` for efficient video data loading and includes an ensemble method where multiple 'tiny' models vote on the final classification, often leading to improved accuracy and robustness.

## Publication

This work is part of a publication. If you use this code, please cite our paper:

- **Paper: (Spanish)** [PDF](publications/pico-JEPA-paper-ES.pdf)
- **Presentation (English):** [PDF](publications/pico-JEPA-presentation-EN.pdf)
- **Presentation (Spanish):** [PDF](publications/pico-JEPA-presentation-ES.pdf)

```bash
@inproceedings{rostagno2025pico,
  title = {pico-JEPA: Comprendiendo el Video con Modelos Ultra-Ligeros y la Sabiduría Colectiva},
  author = {Rostagno, Adrián and Iparraguirre, Javier and Friedrich, Guillermo and Aggio, Santiago and Briatore, Roberto and Tobio, Lucas and Coca, Diego},
  booktitle = {XXXI Congreso Argentino de Ciencias de la Computación (CACIC)},
  year = {2025},
  address = {Viedma, Argentina},
  month = {October},
  note = {6-10 de Octubre}
}
```

## Project Structure

- `app/`: Contains the main application scripts.
  - `train.py`: Script for self-supervised pre-training of the JEPA model.
  - `classify_videos.py`: Script for training a supervised classifier on top of the learned representations.
  - `infer_video.py`: Script for running inference on a single video.
- `visualization/`: Scripts for visualizing model behavior.
  - `visualize_patches_occlusion.py`: Script to visualize the masking strategy used during training.
  - `visualize_features.py`: Script to create a t-SNE plot of the learned features.
- `models/`: Contains the PyTorch model definitions.
- `dataset/`: Contains the `VideoDataset` for data loading.
- `configs/`:
  - `config.yaml`: Configuration file for the general model.
  - `config-[1-4].yaml`: Configuration files for the four 'tiny' models.
- `videos/`: Video dataset.
  - `classInd.txt`: A file containing the index and name of the classes.
- `infers_part.sh`: Shell script to perform inference using the four small models.
- `infers_gral.sh`: Shell script to perform inference using the general model.
- `voting_models.py`: Script for combining the predictions of the four models using a voting ensemble.
- `download_k700.sh`: Shell script to download the dataset.
- `requirements.txt`: Python package dependencies.

---

## Environment Setup (Ubuntu 24.04)

These instructions will guide you through creating a Conda environment and installing the necessary packages.

1. **Create and Activate Conda Environment:**
    Open your terminal and run:

    ```bash
    # Create a new Conda environment named 'pico-jepa' with Python 3.12
    conda create -n pico-jepa python=3.12 -y

    # Activate the newly created environment
    conda activate pico-jepa
    ```

2. **Install PyTorch:**
    Choose one of the following commands based on your hardware:

    - **For systems with a compatible NVIDIA GPU (CUDA support):**

        ```bash
        pip3 install torch torchvision torchaudio
        ```

        *(Note: For specific CUDA versions, visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for the correct command.)*

    - **For CPU-only systems (no NVIDIA GPU):**

        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

3. **Install FFmpeg for `torchcodec`:**
    `torchcodec` relies on FFmpeg. Install a compatible version (e.g., version 6) from `conda-forge` into your environment.

    ```bash
    conda install ffmpeg=6 -c conda-forge -y
    ```

4. **Install `torchcodec`:**
    After FFmpeg is set up, install `torchcodec`.

    ```bash
    pip uninstall torchcodec -y # Uninstall previous attempts if any
    pip install torchcodec --no-cache-dir
    ```

    *(Note: Building `torchcodec` from source might require development tools like CMake, Ninja, a C++ compiler, and NASM. Consult the [official `torchcodec` GitHub](https://github.com/pytorch/torchcodec) if you encounter build issues.)*

5. **Install Remaining Requirements:**

    ```bash
    pip install -r requirements.txt
    ```
---

## How to Run

**Before you start:**

- Download the k700 dataset.

```bash
sh download_k700.sh
```

#### Stage 1: Self-Supervised Pre-training

This stage trains the video encoder using the JEPA methodology.

```bash
python app/train.py 
```

### Stage 2: Video Classification Training General Model

This stage trains the video classification model using the pre-trained encoder.

```bash
python app/classify_videos.py --config_path configs/config.yaml
```

## Stage 3: Video Classification Training Tiny Models

This stage trains the video classification model using the pre-trained encoder for each tiny model configuration.

```bash
python app/classify_videos.py --config_path configs/config-1.yaml
```

```bash
python app/classify_videos.py --config_path configs/config-2.yaml
```

```bash
python app/classify_videos.py --config_path configs/config-3.yaml
```

```bash
python app/classify_videos.py --config_path configs/config-4.yaml
```

### Stage 4: Inference

Infer on the General Model.

```bash
sh infers_gral.sh
```

Infer on Tiny Models.

```bash
sh infers_part.sh
```

### Stage 5: Ensemble Voting for Enhanced Accuracy

To improve classification accuracy, an ensemble method is used. The predictions from the four specialized 'tiny' models are combined through a voting process. This approach can often lead to more robust and accurate results than a single model can provide.

Vote based on the 4 models for each class.

```bash
python voting_models.py
```

Note: The final classification results from the voting are saved in the "vote_results" directory.

### Stage 6: Visualize Patches and Occlusion

```bash
python visualization/visualize_patches_occlusion.py --video_path "videos/infer/abseiling/3E7Jib8Yq5M_000118_000128.mp4"
```

### Stage 7: Visualize Features

```bash
python visualization/visualize_features.py \
    --classifier_path ./pico_jepa_classifier.pth \
    --num_classes 4 \
    --num_videos 20 \
    --max_patches 2000 \
    --class_names "abseiling,adjusting_glasses,alligator_wrestling,archaeological_excavation"
```
