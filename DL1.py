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

    def get_W_shape(self):
        return (self._num_neurons, self._input_shape)

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

    def _trim_sigmoid_backward(self, dA):
        A = self._trim_sigmoid(self._Z)
        dZ = dA * A * (1 - A)
        return dZ

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

    def _trim_tanh_backward(self, dA):
        A = self._trim_tanh(self._Z)
        dZ = dA * (1 - A ** 2)
        return dZ

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
        self._Z = np.dot(self.W, A_prev) + self.b
        A = self.activation_forward(self._Z)
        return A

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]
        self.dW = (1.0 / m) * (dZ @ self._A_prev.T)
        self.db = (1.0 / m) * np.sum(dZ, keepdims=True, axis=1)
        dA_prev = self.W.T @ dZ
        return dA_prev

    def update_parameters(self, dW, db):
        if self._optimization == 'adaptive':
            self._adaptive_alpha_W *= np.where(self._adaptive_alpha_W * dW > 0, self.adaptive_cont,
                                               -self.adaptive_switch)
            self._adaptive_alpha_b *= np.where(self._adaptive_alpha_b * db > 0, self.adaptive_cont,
                                               -self.adaptive_switch)
            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b
        else:
            self.W -= self.alpha * dW
            self.b -= self.alpha * db

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
        s += "\t\tshape weights: (" + str(self._num_neurons) + "," + str(self._input_shape) + ") \n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s


class DLModel:
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False

    def add(self, layer):
        self.layers.append(layer)

    def _squared_means(self, AL, Y):
        J = (AL - Y) ** 2
        return J

    def _squared_means_backward(self, AL, Y):
        dAL = 2 * (AL - Y)
        return dAL

    def _cross_entropy(self, AL, Y):
        J = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return J

    def _cross_entropy_backward(self,AL,Y):
        dAL = np.where(Y == 0, 1/(1-AL), -1/AL)
        return dAL

    def compile(self, loss="squared_means", threshold=0.5):
        self.threshold = threshold
        self.loss = loss
        self._is_compiled = True
        if loss == "squared_means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        elif loss == "cross_entropy":
            self.loss_forward = self._cross_entropy
            self.loss_backward = self._cross_entropy_backward
        elif loss == "categorical_cross_entropy":
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward
        else:
            raise NotImplementedError("Unimplemented loss function: " + loss)

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        errors = self.loss_forward(AL, Y)
        J = (1 / m) * np.sum(errors)
        return J

    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations / 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(1, num_iterations + 1):
            Al = X
            for l in range(1, L):
                Al = self.layers[l].forward_propagation(Al, False)

            if i % print_ind == 0:
                costs.append(self.compute_cost(Al, Y))

            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1, L)):
                dAl, dW, db = self.layers[l].backward_propagation(dAl)
                self.layers[l].update_parameters(dW, db)
        return costs

    def predict(self, X):
        Al = X
        L = len(self.layers)
        for i in range(1, L):
            Al = self.layers[i].forward_propagation(Al, True)

        if Al.shape[0] > 1:  # softmax
            predictions = np.where(Al == Al.max(axis=0), 1, 0)
            return predictions
        else:
            return Al > self.threshold

    def __str__(self):
        s = "Model description:\n\tnum_layers: " + str(len(self.layers) - 1) + "\n"
        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) + "\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1, len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s

