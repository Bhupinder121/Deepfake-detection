import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
#import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image
import seaborn as sns
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import timm

#from keras.models import load_model
#from keras.preprocessing import image
main_path = './data/real_vs_fake/real-vs-fake'

train_dir = './data/real_vs_fake/real-vs-fake/train'
valid_dir = './data/real_vs_fake/real-vs-fake/valid'
test_dir = './data/real_vs_fake/real-vs-fake/test'
images_df = {
    "folder":[],
    "image_path":[],
    "label":[]
}

for folder in os.listdir(main_path): #iterate on each train, valid and test folder
    for label in os.listdir(main_path + "/" + folder): #iterate on fake and real folders (labels)
        for img in glob.glob(main_path + "/" + folder + "/" + label + "/*.jpg"):
            images_df["folder"].append(folder)
            images_df["image_path"].append(img)
            images_df["label"].append(label)
    

images_df = pd.DataFrame(images_df)

real_grouped_df = images_df[images_df['label'] == "real"].groupby('folder')
fake_grouped_df = images_df[images_df['label'] == "fake"].groupby('folder')

train_transforms = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    #transforms.RandomErasing(p=0.3), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



valid_test_transforms = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Keep consistent with training
])
train_ds = ImageFolder(root=train_dir, transform=train_transforms)
valid_ds = ImageFolder(root=valid_dir, transform=valid_test_transforms)
test_ds = ImageFolder(root=test_dir, transform=valid_test_transforms)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True) 
valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)  
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)  
import timm
import torch
import torch.nn as nn

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Xception model (without automatic pretraining)
model = timm.create_model("xception", pretrained=False)

# Load manually downloaded model weights
model_path = "./xception-43020ad28.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

# Get the number of input features for the final FC layer
num_features = model.fc.in_features  # ✅ Change from `classifier` to `fc`

# Replace the FC layer with a binary classification head
model.fc = nn.Linear(num_features, 1)  # ✅ Replace the last layer for binary classification

# Move model to device
model = model.to(device)
'''criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for binary classification
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)'''
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

from accelerate import Accelerator
import torch
from torch.amp import autocast  # ✅ Correct import

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device

# Prepare model, optimizer, and dataloader
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # ✅ Autocast is handled by `Accelerate`, no need for device_type
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), labels)

        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()

        # ✅ Compute accuracy
        predictions = torch.sigmoid(outputs).squeeze(1) > 0.5  # Convert logits to binary predictions
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    # ✅ Compute average loss & accuracy
    avg_loss = total_loss / len(train_loader)
    train_accuracy = correct / total * 100

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")


























