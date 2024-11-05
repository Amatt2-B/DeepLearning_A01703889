import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# Cargar el CSV de metadatos
def load_metadata(csv_path):
    return pd.read_csv(csv_path)

# Generar espectrograma y guardarlo como imagen
def save_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)
    spect = librosa.feature.melspectrogram(y=y, sr=sr)
    spect = librosa.power_to_db(spect, ref=np.max)
    plt.figure(figsize=(2, 2))
    librosa.display.specshow(spect, sr=sr)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Procesar los archivos de audio en espectrogramas
def process_audio_files(metadata, data_dir, output_dir):
    # Crea la carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Procesa cada archivo de audio y guarda el espectrograma
    for index, row in metadata.iterrows():
        audio_path = os.path.join(data_dir, f"fold{row['fold']}", row['slice_file_name'])
        output_path = os.path.join(output_dir, f"{row['slice_file_name'].replace('.wav', '.png')}")
        save_spectrogram(audio_path, output_path)
        print(f"Guardado espectrograma: {output_path}")  # Mensaje de confirmación

# Ejecución principal
if __name__ == "__main__":
    # Define las rutas para el CSV, los archivos de audio y la salida de los espectrogramas
    csv_path = "UrbanSound8K.csv"  # Ruta al archivo CSV
    data_dir = "."  # Ruta donde están las carpetas fold1, fold2, etc.
    output_dir = "spectrograms"  # Carpeta donde se guardarán los espectrogramas

    # Cargar metadatos y procesar archivos de audio
    metadata = load_metadata(csv_path)
    process_audio_files(metadata, data_dir, output_dir)
