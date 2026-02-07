from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0)
