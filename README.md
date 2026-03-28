# Medical Image Analysis System with Transfer Learning and U-Net Segmentation

## Overview

This project implements an end-to-end medical image analysis system that performs both image classification and semantic segmentation using deep learning techniques. The system is designed to assist in medical diagnosis by analyzing chest X-ray images for pneumonia detection and brain MRI scans for tumor segmentation.

The pipeline integrates data preprocessing, model training, evaluation, and visualization into a single reproducible workflow using Docker.

## Goal

The primary objectives of this project are:

- To build a classification model that can detect pneumonia from chest X-ray images using transfer learning.
- To develop a segmentation model that can identify and localize brain tumors in MRI scans using U-Net architecture.
- To create a complete machine learning pipeline that is reproducible, modular, and deployable.
- To generate interpretable outputs including evaluation metrics, predictions, and visualizations.

## Tech Stack

- Programming Language: Python
- Deep Learning Framework: PyTorch
- Computer Vision: OpenCV
- Model Architectures:
  - ResNet50 (Transfer Learning for Classification)
  - U-Net (Segmentation using segmentation-models-pytorch)
- Data Processing: NumPy, scikit-learn
- Visualization: Matplotlib
- Augmentation: Albumentations
- Image Processing: scikit-image
- Containerization: Docker, Docker Compose

## Dataset

- Chest X-Ray Pneumonia Dataset (for classification)
- Brain MRI Segmentation Dataset (for segmentation)

Both datasets are preprocessed into structured training, validation, and test sets.

## Application Flow

The entire system follows a structured pipeline:

1. Data Acquisition
   - Raw datasets are downloaded and stored in `data/raw/`.
   - Includes chest X-ray images and brain MRI scans with masks.

2. Data Preprocessing
   - Images are resized to fixed dimensions:
     - `224x224` for classification
     - `256x256` for segmentation
   - Pixel values are normalized.
   - Data is split into training, validation, and test sets.
   - Processed data is stored in `data/processed/`.

3. Classification Model (Transfer Learning)
   - A pre-trained ResNet50 model is used.
   - Early layers are frozen to retain learned features.
   - Final fully connected layer is modified for binary classification.
   - Model is trained on chest X-ray data.
   - Output: `models/classifier.pth`

4. Segmentation Model (U-Net)
   - U-Net architecture is implemented using `segmentation-models-pytorch`.
   - Model performs pixel-wise classification to detect tumor regions.
   - Dice loss is used for optimization.
   - Output: `models/segmentation.pth`

5. Evaluation Pipeline
   - Classification model is evaluated using:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - Confusion Matrix
   - Segmentation outputs are visualized using:
     - Original image
     - Ground truth mask
     - Predicted overlay
   - Outputs generated:
     - `classification_metrics.json`
     - `classification_predictions.csv`
     - Segmentation images

6. Visualization
   - Training performance curves (loss and accuracy) are plotted and saved in `output/plots/`.

7. Containerized Execution
   - Entire pipeline is containerized using Docker.
   - Running `docker-compose up` executes:
     - Preprocessing
     - Training (classification and segmentation)
     - Evaluation
   - Ensures reproducibility across environments.

## Project Structure

```
medical-image-analysis-system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── output/
│   ├── segmentation_predictions/
│   ├── plots/
│   ├── classification_metrics.json
│   └── classification_predictions.csv
│
├── source_code/
│   ├── preprocess.py
│   ├── train_classifier.py
│   ├── train_segmentation.py
│   ├── evaluate.py
│   └── plot_metrics.py
│
├── Dockerfile
├── docker-compose.yml
├── run_pipeline.sh
├── requirements.txt
├── .env.example
└── README.md
```

## How to Run

### Using Docker (Recommended)

1. Ensure Docker is installed and running.
2. Run:

```
docker-compose up --build
```

3. The pipeline will automatically:
   - Preprocess data
   - Train models
   - Generate outputs

### Without Docker

1. Create a virtual environment:

```
python -m venv venv
```

2. Activate environment and install dependencies:

```
pip install -r requirements.txt
```

3. Run pipeline manually:

```
python source_code/preprocess.py
python source_code/train_classifier.py
python source_code/train_segmentation.py
python source_code/evaluate.py
```

## Outputs

After execution, the following artifacts are generated:

- `models/classifier.pth`
- `models/segmentation.pth`
- `output/classification_metrics.json`
- `output/classification_predictions.csv`
- `output/segmentation_predictions/`
- `output/plots/training_loss_curve.png`
- `output/plots/training_accuracy_curve.png`

## Conclusion

This project demonstrates a complete machine learning workflow for medical image analysis, combining classification and segmentation tasks. It highlights the use of transfer learning, deep learning architectures, and reproducible deployment practices using Docker. The system can be extended further with advanced evaluation techniques and real-time inference capabilities.