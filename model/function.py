import numpy as np


# reference : https://pytorch.org/docs/master/_modules/torch/optim/adam.html
# reference : https://github.com/jrios6/Adam-vs-SGD-Numpy/blob/master/Adam%20vs%20SGD%20-%20On%20Kaggle's%20Titanic%20Dataset.ipynb
class optimizer():
    def __init__(self, lr=1e-4, opt='Adam', betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.iter = 0
        self.lr = lr
        self.opt = opt
        if opt == 'Adam':
            self.iter = 0
            self.beta1, self.beta2 = betas
            self.m = 0
            self.v = 0
            self.eps = eps

    def __call__(self, grad):
        if self.opt == 'Adam':
            # previous addition
            self.iter += 1
            self.m = self.m * self.beta1 + (1 - self.beta1) * grad
            self.v = self.v * self.beta2 + (1 - self.beta2) * (grad ** 2)
            m_corrected = self.m / (1 - (self.beta1 ** self.iter))
            v_corrected = self.v / (1 - (self.beta2 ** self.iter))

            update = m_corrected / (np.sqrt(v_corrected) + self.eps)
            return self.lr * update
        # SGD Parts
        else:
            self.iter += 1
            return self.lr * grad


# flatten --> reshape
def flatten(input):
    batch, input_features, input_height, input_width = input.shape
    # reshape
    output = input.reshape(batch, -1)
    return output


# softmax
def softmax(x, keep_dim=True):
    # to preven overflow
    max = np.max(x, axis=-1)
    max = np.expand_dims(max, axis=-1)
    return np.exp(x - max) / np.sum(np.exp(x - max), axis=-1, keepdims=keep_dim)


# ground truth is hot encode, so we can define softmax with -correct + np.log(---)
def crossEntropy(predict, one_hot):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    correct = predict[one_hot.astype(np.bool)]
    error = -correct + np.log(np.sum(np.exp(predict), axis=1))
    return error


# we can calculate cross entropy grad with math, then softmax(f(x) - p) is grad
def grad_crossEntropy(predict, one_hot):
    # Compute crossentropy gradient from predict[batch,n_classes] and ids of correct answers
    batch_size = predict.shape[0]
    return (-one_hot + softmax(predict)) / batch_size


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_tanh(x):
    return 1 - np.tanh(x) ** 2


def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
