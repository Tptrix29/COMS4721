# This file defines all components which are essential for neural network training, including
# 1. Network layer
# 2. Loss function
# 3. Optimizer
# 4. Dataloader
import numpy as np


class Module:
    """
    Base class for layers.
    """
    def __init__(self):
        self.prev = None  # previous network (linked list of layers)
        self.output = None  # output of forward call for backprop.

    def __call__(self, input):
        if isinstance(input, Module):
            # TODO: chain two networks together with module1(module2(x))
            # update prev and output
            self.output = self.forward(input.output)
        else:
            # TODO: evaluate on an input.
            # update output
            self.output = self.forward(input)
        self.prev = input
        return self

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError


class Sigmoid(Module):
    """
    Sigmoid activation layer class.
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        # TODO: compute sigmoid, update fields
        power = np.clip(-input, a_min=np.finfo(np.float64).minexp, a_max=np.finfo(np.float64).maxexp)
        return 1 / (1 + np.exp(power))

    def backward(self, gradient):
        # TODO: compute gradients with backpropogation and data from forward pass
        return gradient * (1 - self.output) * self.output


class Linear(Module):
    """
    Linear layer class.
    """
    parameter_field = ["weights", "bias"]

    def __init__(self, input_size, output_size, is_input=False):
        super(Linear, self).__init__()
        # TODO: initialize weights and biases.
        self.weights: Parameter = Parameter(np.random.randn(input_size, output_size))
        self.bias: Parameter = Parameter(np.random.randn(output_size))
        self.is_input = is_input

    def forward(self, input):  # input has shape (batch_size, input_size)
        # TODO: compute forward pass through linear input
        return np.matmul(input, self.weights.value) + self.bias.value

    def backward(self, gradient):
        # TODO: compute and store gradients using backpropogation
        _gradient = np.matmul(gradient, self.weights.value.T)
        if self.is_input:
            prev_output = self.prev
        else:
            prev_output = self.prev.output
        # print("Gradient Before: ", self.weights.gradient)
        self.weights.gradient = np.matmul(prev_output.T, gradient)
        self.bias.gradient = np.sum(gradient, axis=0)
        # print("Gradient After: ", self.weights.gradient)
        return _gradient


# generic loss layer for loss functions
class Loss:
    """
    Base class for loss functions.
    """
    def __init__(self):
        self.prev = None

    def __call__(self, input):
        self.prev = input
        return self

    def forward(self, input, labels):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# MSE loss function
class MeanErrorLoss(Loss):
    """
    Mean Squared Error Loss function.
    """
    def __init__(self):
        super(MeanErrorLoss, self).__init__()
        self.gradient = None

    def forward(self, input, labels):  # input has shape (batch_size, input_size)
        # TODO: compute loss, update fields
        n = input.shape[0]
        self.gradient = (input - labels) / n
        return ((input - labels) ** 2).sum() / (2 * n)

    def backward(self):
        # TODO: compute gradient using backpropogation
        return self.gradient


# overall neural network class
class Network(Module):
    """
    Fully-connected neural network class.
    """
    def __init__(self, *args):
        super(Network, self).__init__()
        # TODO: initializes layers, i.e. sigmoid, linear
        current_layer: Module = None
        self.layers = []
        for layer in args:
            if isinstance(layer, Module):
                layer.prev = current_layer
                current_layer = layer
                self.layers.append(layer)
            else:
                raise ValueError("Input must be a Module")

    def __call__(self, input_data):
        return self.forward(input_data)

    def parameters(self):
        parameter_dict = {}
        for layer in self.layers:
            if hasattr(layer, "parameter_field"):
                for attr in layer.__getattribute__("parameter_field"):
                    parameter_dict[str(hash(layer)) + '_' + attr] = layer.__getattribute__(attr)
        return parameter_dict

    def forward(self, input):
        # TODO: compute forward pass through all initialized layers
        x = input
        for layer in self.layers:
            x = layer(x)
        return x.output

    def backward(self, grad):
        # TODO: iterate through layers and compute and store gradients
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

    # def predict(self, data):
    #     # TODO: compute forward pass and output predictions
    #     return self.forward(data)

    # def accuracy(self, test_data, test_labels):
    #     # TODO: evaluate accuracy of model on a test dataset
    #     pred = self.forward(test_data)
    #     return np.sum(np.sum(pred == test_labels, axis=1) == pred.shape[1]) / len(pred)


class Parameter:
    """
    Class to store parameters and gradients for a given layer.
    """
    def __init__(self, value: np.array):
        self.value: np.array = value
        self.gradient: np.array = np.zeros_like(value)

    def __str__(self):
        return f"Parameter: {self.value.shape}"

    def zero_grad(self):
        self.gradient: np.array = np.zeros_like(self.value)


class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, parameters: dict[str, Parameter], lr: float):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for parameter in self.parameters.values():
            # print(f"Before: {parameter.value}")
            parameter.zero_grad()
            # print(f"After: {parameter.value}")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, parameters: dict[str, Parameter], lr: float):
        super(SGD, self).__init__(parameters, lr)

    def step(self):
        for parameter in self.parameters.values():
            parameter.value -= parameter.gradient * self.lr


class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, parameters: dict[str, Parameter], lr: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1E-8):
        super(Adam, self).__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment = {k: np.zeros_like(parameter.value) for (k, parameter) in parameters.items()}
        self.velocity = {k: np.zeros_like(parameter.value) for (k, parameter) in parameters.items()}

    def step(self):
        for (k, parameter) in self.parameters.items():
            moment = self.beta1 * self.moment[k] + (1-self.beta1) * parameter.gradient
            velocity = self.beta2 * self.velocity[k] + (1-self.beta2) * parameter.gradient ** 2
            self.moment[k], self.velocity[k] = moment, velocity
            parameter.value -= self.lr * moment / (np.sqrt(velocity) + self.epsilon)


class DataLoader:
    """
    Data loader class for mini-batch training.
    """
    def __init__(self, X: np.array, y: np.array, batch_size: int, shuffle: bool = False):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = np.arange(X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
        for i in range(0, len(self), self.batch_size):
            yield self[self.idx[i:i + self.batch_size]]

