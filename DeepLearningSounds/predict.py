import torch
from torchvision import transforms
from PIL import Image
import sys
from model import SoundClassifier

# Parámetros de configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo guardado
model = SoundClassifier().to(device)
model.load_state_dict(torch.load("modelo_entrenado_fold10.pth", weights_only=True))
model.eval()

# Transformaciones para el espectrograma
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Ruta del espectrograma de prueba
img_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\adria\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\7mo Sem\DeepLearningSounds\spectrograms\7389-1-1-0.png"

# Cargar y transformar la imagen
image = Image.open(img_path).convert('L')
image = transform(image).unsqueeze(0).to(device)  # Agregar dimensión de batch

# Hacer la predicción
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

# Mapear la predicción a la clase correspondiente
classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", 
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
predicted_class = classes[predicted.item()]

print(f"Predicción de clase: {predicted_class}")
