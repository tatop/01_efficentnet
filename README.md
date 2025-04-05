# EfficientNet Food Classification

A PyTorch implementation of EfficientNet-B0 for food image classification, specifically trained on pizza, steak, and sushi images. This project demonstrates transfer learning using a pre-trained EfficientNet model fine-tuned on a custom food dataset.

## Project Overview

This implementation uses the EfficientNet-B0 architecture with pre-trained weights from ImageNet and fine-tunes it for a specific food classification task. The model is modified to classify three food categories: pizza, steak, and sushi.

## Features

- Transfer learning using pre-trained EfficientNet-B0
- Custom dataset handling and preprocessing
- Training with learning rate optimization
- Real-time inference capabilities
- Training visualization and model performance tracking
- GPU/MPS/CPU support

## Requirements

- Python >= 3.12
- PyTorch >= 2.6.0
- TorchVision >= 0.21.0
- Matplotlib >= 3.10.1
- Other dependencies as listed in pyproject.toml

## Project Structure

```
.
├── data/               # Dataset directory
├── models/             # Saved model weights and training results
├── notebooks/          # Jupyter notebooks for exploration
├── src/
│   ├── data_prep.py    # Data loading and preprocessing
│   ├── engine.py       # Training loop and utilities
│   ├── inference.py    # Inference utilities
│   ├── main.py         # Main training script
│   ├── model.py        # Model architecture
│   └── predict.py      # Prediction utilities
├── pyproject.toml      # Project dependencies and configuration
└── test.py             # Environment and setup testing
```

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install .
   ```
3. Run the test script to verify setup:
   ```bash
   python test.py
   ```
4. Start training:
   ```bash
   python src/main.py
   ```

## Model Architecture

The project uses EfficientNet-B0 as the base model with the following modifications:
- Frozen feature extraction layers
- Custom classifier head with dropout (p=0.2)
- Output layer modified for 3-class classification

## Training

The model is trained with the following specifications:
- Optimizer: AdamW
- Learning Rate: 0.001
- Loss Function: Cross Entropy Loss
- Epochs: 10
- Data Augmentation: Using TorchVision's default EfficientNet transforms

## Results

Training results, including loss and accuracy curves, are automatically saved in the `models` directory. The best model weights are saved as `models/05_efficientnet_model.pth`.

## Inference

To perform inference on new images, use the `predict.py` script:
```python
from src.predict import pred_and_plot_image
pred_and_plot_image(model, image_path, class_names, transform)
```

## Development

For development work, additional tools are available:
- Ruff for code formatting and linting
- Notebook support for experimentation

## License

This project is open-source and available for educational and research purposes.
