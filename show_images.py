import numpy as np
import gzip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load images and labels ──────────────────
def load_data():
    with gzip.open("mnist_data/t10k-images-idx3-ubyte.gz", "rb") as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

    with gzip.open("mnist_data/t10k-labels-idx1-ubyte.gz", "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels

# ── Load your trained model ─────────────────
from neural_network import NeuralNetwork

def load_model():
    data = np.load("model_weights.npy", allow_pickle=True).item()
    nn = NeuralNetwork([784, 128, 64, 10])
    nn.weights = data["weights"]
    nn.biases  = data["biases"]
    return nn

# ── Show grid of images with predictions ───
def show_predictions(images, labels, nn, num=40):
    fig, axes = plt.subplots(5, 8, figsize=(16, 10))
    fig.suptitle("MNIST — Your Neural Network Predictions", 
                 fontsize=16, fontweight='bold', y=1.02)

    for i, ax in enumerate(axes.flat):
        img = images[i]
        true_label = labels[i]

        # Get prediction
        pixel_input = img.flatten().reshape(1, -1) / 255.0
        probs = nn.forward(pixel_input)[0]
        predicted = np.argmax(probs)
        confidence = probs[predicted] * 100

        # Show image
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        # Green title = correct, Red = wrong
        color = 'green' if predicted == true_label else 'red'
        ax.set_title(
            f"Pred: {predicted} ({confidence:.0f}%)\nTrue: {true_label}",
            fontsize=8,
            color=color,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150, bbox_inches='tight')
    print("✅ Saved as predictions.png")
    plt.show()

# ── Show digits 0-9 one of each ────────────
def show_one_of_each(images, labels, nn):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("One Example of Each Digit (0–9)", 
                 fontsize=14, fontweight='bold')

    for digit in range(10):
        idx = np.where(labels == digit)[0][0]
        img = images[idx]

        pixel_input = img.flatten().reshape(1, -1) / 255.0
        probs = nn.forward(pixel_input)[0]
        predicted = np.argmax(probs)
        confidence = probs[predicted] * 100

        ax = axes[digit // 5][digit % 5]
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        color = 'green' if predicted == digit else 'red'
        ax.set_title(
            f"Digit: {digit}\nPred: {predicted} ({confidence:.0f}%)",
            fontsize=10,
            color=color,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig("digits_0_to_9.png", dpi=150, bbox_inches='tight')
    print("✅ Saved as digits_0_to_9.png")
    plt.show()

# ── Show worst mistakes ─────────────────────
def show_mistakes(images, labels, nn, num=16):
    mistakes = []
    for i in range(len(images)):
        pixel_input = images[i].flatten().reshape(1, -1) / 255.0
        probs = nn.forward(pixel_input)[0]
        predicted = np.argmax(probs)
        confidence = probs[predicted] * 100
        if predicted != labels[i]:
            mistakes.append((i, predicted, labels[i], confidence))
        if len(mistakes) == num:
            break

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("❌ Where Your Network Was Wrong", 
                 fontsize=14, fontweight='bold', color='red')

    for j, (idx, pred, true, conf) in enumerate(mistakes):
        ax = axes[j // 4][j % 4]
        ax.imshow(images[idx], cmap='Reds')
        ax.axis('off')
        ax.set_title(
            f"Guessed: {pred} ({conf:.0f}%)\nActual: {true}",
            fontsize=9, color='darkred', fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig("mistakes.png", dpi=150, bbox_inches='tight')
    print("✅ Saved as mistakes.png")
    plt.show()


# ── RUN ALL ─────────────────────────────────
if __name__ == "__main__":
    print("Loading data and model...")
    images, labels = load_data()
    nn = load_model()

    print("\n📊 Showing 40 predictions...")
    show_predictions(images, labels, nn)

    print("\n🔢 Showing one of each digit...")
    show_one_of_each(images, labels, nn)

    print("\n❌ Showing network mistakes...")
    show_mistakes(images, labels, nn)

    print("\nDone! Check your folder for 3 saved images.")