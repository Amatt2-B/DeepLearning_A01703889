import torch
from model import SoundClassifier

# Crear una instancia del modelo
model = SoundClassifier()

# Crear un tensor aleatorio con forma (batch_size, canales, altura, ancho)
# En este caso, batch_size = 1, canales = 1, altura = 64, ancho = 64
dummy_input = torch.randn(1, 1, 64, 64)

# Pasar el tensor a través del modelo
output = model(dummy_input)

# Verificar la salida
print("Forma de la salida:", output.shape)  # Debería ser (1, 10) ya que hay 10 clases
