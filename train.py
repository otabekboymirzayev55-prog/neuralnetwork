import numpy as np
import urllib.request
import gzip
import os
from neural_network import NeuralNetwork

# ──────────────────────────────────────────
# 1. LOAD MNIST (no Keras, no TensorFlow)
# ──────────────────────────────────────────
def download_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs("mnist_data", exist_ok=True)
    for name, fname in files.items():
        path = f"mnist_data/{fname}"
        if not os.path.exists(path):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, path)
    print("All files ready.\n")

def load_mnist():
    download_mnist()

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            f.read(16)  # skip header
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 784) / 255.0

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            f.read(8)   # skip header
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images("mnist_data/train-images-idx3-ubyte.gz")
    y_train = read_labels("mnist_data/train-labels-idx1-ubyte.gz")
    X_test  = read_images("mnist_data/t10k-images-idx3-ubyte.gz")
    y_test  = read_labels("mnist_data/t10k-labels-idx1-ubyte.gz")

    Y_train = np.eye(10)[y_train]  # one-hot encode
    Y_test  = np.eye(10)[y_test]

    return X_train, Y_train, X_test, Y_test, y_test


# ──────────────────────────────────────────
# 2. TRAINING LOOP
# ──────────────────────────────────────────
def train(nn, X_train, Y_train, epochs=20, batch_size=64, lr=0.01):
    m = X_train.shape[0]
    for epoch in range(epochs):
        indices   = np.random.permutation(m)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        epoch_loss  = 0
        num_batches = 0

        for start in range(0, m, batch_size):
            X_batch = X_shuffled[start:start+batch_size]
            Y_batch = Y_shuffled[start:start+batch_size]

            Y_pred = nn.forward(X_batch)
            loss   = nn.compute_loss(Y_pred, Y_batch)
            gW, gb = nn.backward(Y_batch)
            nn.update_weights(gW, gb, lr)

            epoch_loss  += loss
            num_batches += 1

        print(f"Epoch {epoch+1:2d}/{epochs}  loss: {epoch_loss/num_batches:.4f}")


# ──────────────────────────────────────────
# 3. ACCURACY
# ──────────────────────────────────────────
def accuracy(nn, X_test, y_test_labels):
    probs       = nn.forward(X_test)
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions == y_test_labels)


# ──────────────────────────────────────────
# 4. RUN
# ──────────────────────────────────────────
if __name__ == "__main__":
    print("Loading MNIST...")
    X_train, Y_train, X_test, Y_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    nn = NeuralNetwork([784, 128, 64, 10])

    print("Training...\n")
    train(nn, X_train, Y_train, epochs=20, batch_size=64, lr=0.01)

    acc = accuracy(nn, X_test, y_test)
    print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%")
    # Save only after a successful explicit training run.
    np.save("model_weights.npy", {"weights": nn.weights, "biases": nn.biases})
    print("Model saved to model_weights.npy")
