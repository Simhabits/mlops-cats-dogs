from PIL import Image
import numpy as np
from src.utils.preprocess import preprocess_image

def test_preprocess_image_shape():
    img = Image.new("RGB", (256, 256))
    tensor = preprocess_image(img)

    assert tensor.shape == (1, 3, 224, 224)
