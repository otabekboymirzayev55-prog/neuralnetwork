# MNIST Neural Network from Scratch

A feedforward neural network built using **only NumPy** (no PyTorch or TensorFlow).

## Results
- 🎯 **96.46% accuracy** on the MNIST test set
- Trained on 60,000 handwritten digit images

## Architecture
784 → 128 → 64 → 10

- ReLU activation on hidden layers
- Softmax on output layer
- Cross-entropy loss
- Mini-batch SGD optimizer

## Files
| File | Description |
|---|---|
| `neural_network.py` | Neural net class with backpropagation |
| `train.py` | Training loop + MNIST loader |
| `agent_tool.py` | Wrapped as an AI agent tool |
| `show_images.py` | Visualize predictions |

## How to Run
pip install numpy matplotlib
python train.py
python show_images.py
