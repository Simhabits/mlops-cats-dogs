from PIL import Image
from torchvision import transforms

# EXACT SAME AS TRAINING
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0)
