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

# Definir las transformaciones para los espectrogramas
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class SpectrogramDataset(Dataset):
    def __init__(self, metadata, img_dir, transform=None):
        self.metadata = metadata
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['slice_file_name'].replace('.wav', '.png')  # Cambiar la extensión a .png
        label = row['classID']
        
        # Ruta completa del espectrograma
        img_path = os.path.join(self.img_dir, img_name)
        
        # Cargar la imagen
        try:
            image = Image.open(img_path).convert('L')  # Convertir a escala de grises
        except FileNotFoundError:
            print(f"Error: Archivo no encontrado - {img_path}")
            raise

        # Aplicar transformaciones si existen
        if self.transform:
            image = self.transform(image)

        return image, label


# Función de entrenamiento
def train(model, train_loader, criterion, optimizer):
    model.train()  # Configura el modelo en modo de entrenamiento
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Reinicia los gradientes
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass y optimización
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)

# Función de evaluación
def evaluate(model, val_loader, criterion):
    model.eval()  # Configura el modelo en modo de evaluación
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # Calcular precisión
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(val_loader.dataset)
    return running_loss / len(val_loader.dataset), accuracy

# Validación cruzada
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
        
        # Guardar el modelo al final del fold actual
        torch.save(model.state_dict(), f"modelo_entrenado_fold{fold+1}.pth")
        
        results.append((val_loss, val_accuracy))
    
    return results

# Parámetros de configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.0005
epochs = 15
