import os
import zipfile
import requests
import torch
import torchvision
from pathlib import Path
from torch import nn
import argparse
from data_prep import create_data_loader
from engine import train
from model import EfficientNetScaled
from predict import pred_and_plot_image
import random
from timeit import default_timer as timer 

# Argument parsing
parser = argparse.ArgumentParser(description='Train EfficientNet model for food classification')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train (default: 15)')
parser.add_argument('--phi', type=int, default=0, help='EfficientNet scaling parameter (default: 0)')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate (default: 0.01)')
args = parser.parse_args()

#device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
print(f"Using device: {device}")
#torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"


# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi_20_percent.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi_20_percent.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)

    # Remove .zip file
    os.remove(data_path / "pizza_steak_sushi_20_percent.zip")

train_dir = image_path / "train"
test_dir = image_path / "test"

print(f"[INFO] Pizza steak sushi data downloaded and extracted to {image_path}")
print(f"[INFO] Train data: {train_dir}")
print(f"[INFO] Test data: {test_dir}")

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
transform = weights.transforms()

train_dl, test_dl, class_names = create_data_loader(train_dir=train_dir,
                                                    test_dir=test_dir,
                                                    transform=transform,
                                                    num_workers=0)

model_b0 = EfficientNetScaled(phi=args.phi, num_classes=3).to(device)

"""
# Create a model with similar architecture to the pretrained EfficientNet model
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

for param in model.features.parameters():
    param.requires_grad = False

output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)
"""

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model_b0.parameters(), lr=args.lr)  # LR iniziale più alto

# Calculate total steps for scheduler
total_steps = len(train_dl) * args.epochs

# Create OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,  # learning rate massimo
    total_steps=total_steps,
    pct_start=0.1,  # 10% degli step per il warm-up
    anneal_strategy='cos'  # annealing cosinusoidale
)

# Start the timer
start_time = timer()

if __name__ == '__main__':
    # Setup training and save the results
    results = train(model=model_b0,
                   train_dataloader=train_dl,
                   test_dataloader=test_dl,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   loss_fn=loss_fn,
                   epochs=args.epochs,
                   device=device,
                   patience=args.patience)

    
    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True, parents=True)

     # Salvataggio risultati di training
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(results["train_loss"])), results["train_loss"], label="Train loss")
    plt.plot(range(len(results["test_loss"])), results["test_loss"], label="Test loss")
    plt.title("Loss curves")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(results["train_acc"])), results["train_acc"], label="Train accuracy")
    plt.plot(range(len(results["test_acc"])), results["test_acc"], label="Test accuracy")
    plt.title("Accuracy curves")
    plt.legend()
    plt.savefig(model_dir / "training_results.png")

    num_images_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
    test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                        k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model_b0, 
                            image_path=image_path,
                            class_names=class_names,
                            transform=transform)
        

    # Save the model
    model_save_path = "models/05_efficientnet_model_b0.pth"
    torch.save(obj=model_b0.state_dict(), f=model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")