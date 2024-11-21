from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import librosa
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from model import SoundClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clases de sonido
classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", 
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

# Cargar el modelo
model = SoundClassifier().to(device)
model.load_state_dict(torch.load("modelo_entrenado_fold1.pth", map_location=device))
model.eval()

# Transformación para el espectrograma
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Crear y guardar el espectrograma
    plt.figure(figsize=(2, 2))
    plt.axis('off')
    plt.imshow(spectrogram_db, aspect='auto', origin='lower')
    plt.savefig('temp_spectrogram.png', bbox_inches='tight', pad_inches=0)
    plt.close()  # Asegúrate de cerrar la figura de matplotlib

    # Cargar la imagen usando PIL y transformarla
    image = Image.open('temp_spectrogram.png').convert('L')
    image = transform(image).unsqueeze(0).to(device)
    return image

def classify_audio(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se proporcionó ningún archivo"}), 400

    # Guardar archivo de audio
    file = request.files['file']
    filename = secure_filename(file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(audio_path)

    # Generar y clasificar el espectrograma
    image = audio_to_spectrogram(audio_path)
    predicted_class = classify_audio(image)

    # Borrar el espectrograma temporal pero mantener el audio para el reproductor
    os.remove('temp_spectrogram.png')

    return render_template("result.html", prediction=predicted_class, audio_file=filename)

# Ruta para servir archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
