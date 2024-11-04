# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundClassifier(nn.Module):
    def __init__(self):
        super(SoundClassifier, self).__init__()
        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 canal (escala de grises)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Capas densas (fully connected) - ajustadas según la forma de x
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Cambiado de 16 * 16 a 8 * 8
        self.fc2 = nn.Linear(256, 10)  # 10 clases de sonido

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Cambiado de 16 * 16 a 8 * 8
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
