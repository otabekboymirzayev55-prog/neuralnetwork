import numpy as np
from neural_network import NeuralNetwork

# ──────────────────────────────────────────
# LOAD TRAINED MODEL ONCE (at import time)
# ──────────────────────────────────────────
def _load_model():
    data = np.load("model_weights.npy", allow_pickle=True).item()
    nn = NeuralNetwork([784, 128, 64, 10])
    nn.weights = data["weights"]
    nn.biases  = data["biases"]
    return nn

_model = _load_model()

# ──────────────────────────────────────────
# THE AGENT TOOL FUNCTION
# ──────────────────────────────────────────
def predict_digit(image_input):
    """
    Agent tool: classify a handwritten digit image.

    Args:
        image_input: one of:
            - flat list/array of 784 numbers (pixel values 0-255)
            - 28x28 nested list or numpy array

    Returns:
        dict with keys:
            - predicted_digit (int): 0-9
            - confidence (float): 0.0-1.0
            - all_probabilities (dict): probability for each digit
    """
    arr = np.array(image_input, dtype=float)

    # Handle both flat and 2D input
    if arr.shape == (28, 28):
        arr = arr.flatten()

    if arr.shape != (784,):
        return {"error": f"Expected 784 pixels, got {arr.shape}"}

    # Normalize if values look like 0-255
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Run through network
    probs = _model.forward(arr.reshape(1, -1))[0]

    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted])

    return {
        "predicted_digit":   predicted,
        "confidence":        round(confidence, 4),
        "all_probabilities": {
            str(i): round(float(p), 4) for i, p in enumerate(probs)
        }
    }


# ──────────────────────────────────────────
# TOOL DEFINITION (LangChain / CrewAI style)
# ──────────────────────────────────────────
TOOL_DEFINITION = {
    "name": "predict_digit",
    "description": (
        "Classifies a handwritten digit image (0-9). "
        "Input: a flat array of 784 pixel values (0-255) "
        "representing a 28x28 grayscale image. "
        "Returns the predicted digit and confidence score."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_input": {
                "type":        "array",
                "items":       {"type": "number"},
                "description": "784 pixel values (0-255), row by row"
            }
        },
        "required": ["image_input"]
    }
}


# ──────────────────────────────────────────
# TEST THE TOOL
# ──────────────────────────────────────────
if __name__ == "__main__":
    import gzip

    # Load one real MNIST image to test
    with gzip.open("mnist_data/t10k-images-idx3-ubyte.gz", "rb") as f:
        f.read(16)
        pixels = np.frombuffer(f.read(784), dtype=np.uint8)

    with gzip.open("mnist_data/t10k-labels-idx1-ubyte.gz", "rb") as f:
        f.read(8)
        true_label = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])

    print(f"True label:  {true_label}")
    result = predict_digit(pixels.tolist())
    print(f"Prediction:  {result['predicted_digit']}")
    print(f"Confidence:  {result['confidence'] * 100:.1f}%")
    print(f"\nAll probabilities:")
    for digit, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 40)
        print(f"  {digit}: {bar} {prob:.4f}")