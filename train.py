# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
from model import SoundClassifier
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = 0.001
epochs = 10

# Dataset personalizado para cargar espectrogramas
class SpectrogramDataset(Dataset):
    def __init__(self, csv_metadata, img_dir, transform=None):
        self.metadata = csv_metadata
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['slice_file_name'].replace('.wav', '.png'))
        image = Image.open(img_name).convert('L')  # Convertir a escala de grises
        label = int(self.metadata.iloc[idx]['classID'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformaciones de datos
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Función de entrenamiento
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Función de evaluación
def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    return val_loss / len(val_loader), correct / len(val_loader.dataset)

# 10-Fold Cross-Validation
def cross_validate(model_class, csv_metadata, img_dir):
    kfold = KFold(n_splits=10)
    results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(csv_metadata)):
        train_subset = csv_metadata.iloc[train_idx]
        val_subset = csv_metadata.iloc[val_idx]

        train_dataset = SpectrogramDataset(train_subset, img_dir, transform=transform)
        val_dataset = SpectrogramDataset(val_subset, img_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print(f"\nEntrenando en Fold {fold+1}/{kfold.get_n_splits()}...")
        for epoch in range(epochs):
            train_loss = train(model, train_loader, criterion, optimizer)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion)
            print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
        
        results.append((val_loss, val_accuracy))
    
    return results
