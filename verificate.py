import pandas as pd

# Cargar el archivo CSV
metadata = pd.read_csv('UrbanSound8k.csv')

# Nombre del archivo de audio original (sin la extensión de espectrograma)
audio_file_name = '6988-5-0-1.wav'

# Filtrar el DataFrame para encontrar el archivo y su clase
file_info = metadata[metadata['slice_file_name'] == audio_file_name]

# Mostrar la clase real
if not file_info.empty:
    real_class = file_info.iloc[0]['class']
    print(f"Clase real para {audio_file_name}: {real_class}")
else:
    print(f"Archivo {audio_file_name} no encontrado en los metadatos.")
