import numpy as np
import matplotlib.pyplot as plt
import random


class DLLayer:
    def __init__(self, num_neurons, input_shape, activation="relu", W_initialization="random", learning_rate=0.01,
                 optimization="adaptive"):
        self._num_neurons = num_neurons
        self._input_shape = input_shape
        self._activation = activation
        self._optimization = optimization
        self.alpha = float(learning_rate)
        self.random_scale = 0.01
        self.init_weights(W_initialization)

        if self._optimization == 'adaptive':
            self._adaptive_alpha_b = np.full((self._num_neurons, 1), self.alpha)
            self._adaptive_alpha_W = np.full(self.get_W_shape(), self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = 0.5

        self.activation_trim = 1e-10
        if activation == "leaky_relu":
            self.leaky_relu_d = 0.01

        if activation == "sigmoid":
            self.activation_forward = self._sigmoid
            self.activation_backward = self._sigmoid_backward
        if activation == "trim_sigmoid":
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._trim_sigmoid_backward
        if activation == "trim_tanh":
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._trim_tanh_backward
        if activation == "tanh":
            self.activation_forward = self._tanh
            self.activation_backward = self._tanh_backward
        elif activation == "relu":
            self.activation_forward = self._relu
            self.activation_backward = self._relu_backward
        elif activation == "leaky_relu":
            self.activation_forward = self._leaky_relu
            self.activation_backward = self._leaky_relu_backward

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_neurons, 1), dtype=float)
        if W_initialization == "zeros":
            self.W = np.full(self.get_W_shape(), self.alpha)
        elif W_initialization == "random":
            self.W = np.random.randn(*self.get_W_shape()) * self.random_scale
        elif W_initialization == "He":
            self.W = np.random.randn(*self.get_W_shape()) * np.sqrt(2.0 / sum(self._input_shape))
        elif W_initialization == "Xaviar":
            self.W = np.random.randn(*self.get_W_shape()) * np.sqrt(1.0 / sum(self._input_shape))

    def get_W_shape(self):
        return self._num_neurons, *self._input_shape

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        dZ = dA * A * (1 - A)
        return dZ

    def _trim_sigmoid(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1 / (1 + np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = A = 1 / (1 + np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_backward(self, dA):
        A = self._tanh(self._Z)
        dZ = dA * (1 - A ** 2)
        return dZ

    def _trim_tanh(self, Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, 0, dA)
        return dZ

    def _leaky_relu(self, Z):
        return np.where(Z > 0, Z, self.leaky_relu_d * Z)

    def _leaky_relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, self.leaky_relu_d * dA, dA)
        return dZ

    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = self.W @ A_prev + self.b
        A = self.activation_forward(self._Z)
        return A

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]
        self.dW = (1.0 / m) * (dZ @ self._A_prev.T)
        self.db = (1.0 / m) * np.sum(dZ, keepdims=True, axis=1)
        dA_prev = self.W.T @ dZ
        return dA_prev

    def update_parameters(self):
        if self._optimization == 'adaptive':
            self._adaptive_alpha_W *= np.where(self._adaptive_alpha_W * self.dW > 0, self.adaptive_cont,
                                               -self.adaptive_switch)
            self._adaptive_alpha_b *= np.where(self._adaptive_alpha_b * self.db > 0, self.adaptive_cont,
                                               -self.adaptive_switch)
            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b
        else:
            self.W -= self.alpha * self.dW
            self.b -= self.alpha * self.db

    def __str__(self):
        s = "Perseptrons Layer:\n"
        s += "\tnum_neurons: " + str(self._num_neurons) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d) + "\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        # optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont) + "\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch) + "\n"
        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.get_W_shape) + "\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s
