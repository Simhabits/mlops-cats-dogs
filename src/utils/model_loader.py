import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(16 * 31 * 31, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_model(model_path: str):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
