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

def preprocess_image(image: Image.Image, model=None) -> torch.Tensor:
    """
    Preprocess image to match model input features.
    - If model is provided, adaptively pool to match classifier in_features.
    - If mismatch occurs, auto‑patch classifier layer.
    - If no model is provided, just return resized + normalized tensor.
    """
    x = transform(image).unsqueeze(0)  # [1, 3, H, W]

    if model is not None:
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
        spatial = int((in_features / x.shape[1]) ** 0.5)
        h, w = spatial, spatial
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