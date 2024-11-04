# main.py
import pandas as pd
from train import cross_validate
from model import SoundClassifier
from data_processing import load_metadata

# Configuración de rutas
csv_path = "UrbanSound8K.csv"  # Ruta del archivo CSV
img_dir = "spectrograms"  # Carpeta de espectrogramas generados

# Cargar metadatos y realizar entrenamiento y validación cruzada
metadata = load_metadata(csv_path)
results = cross_validate(SoundClassifier, metadata, img_dir)

# Mostrar resultados promedio
avg_loss = sum([result[0] for result in results]) / len(results)
avg_accuracy = sum([result[1] for result in results]) / len(results)
print(f"\nResultados promedio - Pérdida: {avg_loss:.4f}, Precisión: {avg_accuracy:.4f}")
