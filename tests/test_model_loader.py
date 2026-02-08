from src.utils.model_loader import load_model

def test_model_loads():
    model = load_model("output/model.pt")
    assert model is not None

