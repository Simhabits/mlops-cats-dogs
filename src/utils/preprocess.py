from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Standard transform (resize + normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def compute_pooling_dims(in_features: int, channels: int = 3):
    """
    Compute pooling dimensions to approximately match model input size.
    """
    spatial = int((in_features / channels) ** 0.5)
    return spatial, spatial

def preprocess_image(image: Image.Image, model) -> torch.Tensor:
    """
    Preprocess image to match model input features.
    If mismatch occurs, auto‑patch the classifier layer.
    """
    x = transform(image).unsqueeze(0)  # [1, 3, H, W]

    # Get expected feature size from model
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        classifier_layer = model.fc
    elif hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        classifier_layer = model.classifier
    else:
        raise AttributeError("Model does not have 'fc' or 'classifier' attribute")

    # Compute pooling dimensions
    h, w = compute_pooling_dims(in_features, channels=x.shape[1])
    x = F.adaptive_avg_pool2d(x, (h, w))

    flattened_size = x.view(x.size(0), -1).shape[1]
    print(f"Pooling to: {(h, w)} → Flattened size: {flattened_size}")

    # Auto‑patch classifier if mismatch
    if flattened_size != in_features:
        print(f"⚠️ Mismatch detected: classifier expects {in_features}, got {flattened_size}")
        print("🔧 Patching classifier layer automatically...")
        classifier_layer = nn.Linear(flattened_size, classifier_layer.out_features)
        if hasattr(model, "fc"):
            model.fc = classifier_layer
        else:
            model.classifier = classifier_layer
        print(f"✅ Classifier patched: now expects {flattened_size}")

    return x