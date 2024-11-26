import pandas as pd
import torch
from train import cross_validate
from model import SoundClassifier
from data_processing import load_metadata

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

csv_path = "UrbanSound8K.csv"
img_dir = "spectrograms"

# Realizar entrenamiento y validación cruzada, pasando el dispositivo
metadata = load_metadata(csv_path)
results = cross_validate(SoundClassifier, metadata, img_dir, device=device)

# Mostrar resultados promedio
avg_loss = sum([result[0] for result in results]) / len(results)
avg_accuracy = sum([result[1] for result in results]) / len(results)
print(f"\nResultados promedio - Pérdida: {avg_loss:.4f}, Precisión: {avg_accuracy:.4f}")
