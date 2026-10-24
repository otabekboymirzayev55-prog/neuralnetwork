import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            n_in  = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def _softmax(self, Z):
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.Z_values    = []
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            self.Z_values.append(Z)
            is_last_layer = (i == len(self.weights) - 1)
            A = self._softmax(Z) if is_last_layer else self._relu(Z)
            self.activations.append(A)
        return self.activations[-1]

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        log_probs = -np.log(Y_pred + 1e-8)
        loss = np.sum(Y_true * log_probs) / m
        return loss

    def backward(self, Y_true):
        m = Y_true.shape[0]
        grads_W = []
        grads_b = []
        dA = self.activations[-1] - Y_true
        for i in reversed(range(len(self.weights))):
            A_prev = self.activations[i]
            dW = (A_prev.T @ dA) / m
            db = np.sum(dA, axis=0, keepdims=True) / m
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            if i > 0:
                dA = (dA @ self.weights[i].T) * self._relu_derivative(self.Z_values[i - 1])
        return grads_W, grads_b

    def update_weights(self, grads_W, grads_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_W[i]
            self.biases[i]  -= learning_rate * grads_b[i]


# ── SANITY TEST ─────────────────────────────
if __name__ == "__main__":
    nn = NeuralNetwork([784, 128, 64, 10])

    X_fake = np.random.randn(5, 784)
    Y_fake = np.eye(10)[np.random.randint(0, 10, 5)]

    probs = nn.forward(X_fake)
    print("Output shape:", probs.shape)        # → (5, 10)
    print("Row sums:", probs.sum(axis=1))      # → all ~1.0

    loss = nn.compute_loss(probs, Y_fake)
    print("Initial loss:", round(loss, 4))     # → ~2.3

    gW, gb = nn.backward(Y_fake)
    nn.update_weights(gW, gb, learning_rate=0.01)
    print("✅ Backprop and update worked!")