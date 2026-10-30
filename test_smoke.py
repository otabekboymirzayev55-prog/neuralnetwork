import unittest
import numpy as np
import train
import agent_tool
from neural_network import NeuralNetwork

class NetworkTests(unittest.TestCase):
    def test_probabilities_and_update(self):
        np.random.seed(7)
        nn = NeuralNetwork([4, 5, 2])
        x = np.ones((3, 4))
        y = np.eye(2)[[0, 1, 0]]
        p = nn.forward(x)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        before = nn.weights[0].copy()
        nn.update_weights(*nn.backward(y), 0.1)
        self.assertFalse(np.array_equal(before, nn.weights[0]))
    def test_bad_pixels(self):
        self.assertIn('error', agent_tool.predict_digit([float('nan')] * 784))
        self.assertIn('error', agent_tool.predict_digit(['bad'] * 784))
        self.assertIn('error', agent_tool.predict_digit([0]))
if __name__ == '__main__': unittest.main()
