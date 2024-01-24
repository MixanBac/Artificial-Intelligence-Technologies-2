import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import os
import shutil

# Описание:
# В данной программе используется простая архитектура с двумя сверточными слоями.
# Первый сверточный слой имеет 64 канала с размером ядра 3x3, а второй сверточный слой преобразует 64 канала обратно в 3 канала.
# Между сверточными слоями используется функция активации ReLU для введения нелинейности.

class ImageProcessingModel(nn.Module):
    def __init__(self):
        super(ImageProcessingModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = ToTensor()
    image = transform(image).unsqueeze(0)
    return image

def process_image(image_path, output_path):
    model = ImageProcessingModel()
    image = preprocess_image(image_path)
    output = model(image)

    output_image = output.squeeze().detach().numpy().transpose((1, 2, 0))
    output_image = (output_image * 255).astype('uint8')
    output_image = Image.fromarray(output_image)
    output_image.save(output_path)

image_path = "t.jpg"
output_path = "res.jpg"
process_image(image_path, output_path)
print('Успех!')