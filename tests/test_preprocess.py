from PIL import Image
from src.utils.preprocess import preprocess_image

def test_preprocess_image_shape():
    img = Image.new("RGB", (256, 256))
    tensor = preprocess_image(img)

    # Must match training preprocessing
    assert tensor.shape == (1, 3, 64, 64)
