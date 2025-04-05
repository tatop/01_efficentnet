from pathlib import Path
import random
import torch
from torch import nn
import torchvision
from model import EfficientNetScaled
from predict import pred_and_plot_image

device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

test_dir = "data/pizza_steak_sushi/test"

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
transform = weights.transforms()
class_names = ["pizza", "steak", "sushi"]

def create_effnetb0():
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(42)

    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=3)
    ).to(device)

    # 5. Give the model a name
    model.name = "effnetb0"
    print(f"[INFO] Created new {model.name} model.")
    return model

# Load the model
best_model_path = "models/05_efficientnet_model_b0.pth"
# model = create_effnetb0()
model = EfficientNetScaled(phi=0, num_classes=3).to(device)
model.load_state_dict(torch.load(best_model_path))


# Load the image

num_images_to_plot = 3
test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                        k=num_images_to_plot) # randomly select 'k' image paths to pred and plot



for image_path in test_image_path_sample:
    pred_and_plot_image(model=model, 
                        image_path=image_path,
                        class_names=class_names,
                        transform=transform)
    

pred_and_plot_image(model=model, 
                    image_path="pizza.jpg",
                    class_names=class_names,
                    transform=transform)