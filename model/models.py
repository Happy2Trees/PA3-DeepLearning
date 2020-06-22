import pickle

import numpy as np

import model.submodule as nn


# wrapping class for forward and backward for sequences
class SeqModule(object):
    def __init__(self, sequential):
        self.activation = list()
        self.sequential = sequential

    def __call__(self, x):
        self.activation = list()
        self.activation.append(x)
        for layer in self.sequential:
            x = layer(x)
            self.activation.append(x)
        self.activation.pop(-1)
        return x

    def backward(self, grad_output):
        rev_activation = list(reversed(self.activation))
        for index, layer in enumerate(reversed(self.sequential)):
            grad_output = layer.backward(rev_activation[index], grad_output)

        return grad_output


# for LSTM Image Captioning
class ImageCaptionLSTM(object):
    def __init__(self, embed, dropout=False):
        super().__init__()
        self.CNNSequential = SeqModule([
            # convolution 1 + Relu
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            # convolution 2 + Relu + pooling
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # convolution 3 + Relu + pooling
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # convolution 4 + Relu + pooling
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        # 5th layer
        self.CNN2Sequential = SeqModule([
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
        ])
        ############
        # Concatenate
        ############
        self.afterCNN = SeqModule([
            nn.MaxPool2d(2),
            # same as reshape
            nn.flatten(),
            nn.Dropout(check=dropout),
            nn.linear(1536, 256),
            nn.tanh()
        ])
        self.Dropout = nn.Dropout(check=dropout)
        self.LSTM = SeqModule([
            nn.LSTM(embed, 256),
            nn.tanh(),
        ])
        ############
        # 중간에 ADD
        ############

        self.lastLayer = SeqModule([
            nn.linear(256, 256),
            nn.ReLU(),
            nn.linear(256, 1665)
        ])

    def __call__(self, images, embed):
        return self.forward(images, embed)

    # forward each layer
    def forward(self, images, embed):
        # CNN Encoder
        x1 = self.CNNSequential(images)
        x2 = self.CNN2Sequential(x1)
        x = np.concatenate((x1, x2), axis=1)
        x = self.afterCNN(x)

        # RNN Encoder
        y = self.Dropout(embed)
        y = self.LSTM(y)
        seq_len, batch, _ = y.shape

        z = np.zeros(y.shape)
        output = np.zeros((seq_len, batch, 1665))

        for i in range(seq_len):
            # ADD
            z[i] = x + y[i]
            # Last Layer of Networks
            output[i] = self.lastLayer(z[i])
        return output

    # update parameters
    def update(self, grad_output):
        seq_len, batch, _ = grad_output.shape
        dz = np.zeros((seq_len, batch, 256))

        for i in reversed(range(seq_len)):
            dz[i] = self.lastLayer.backward(grad_output[i])

            # CNN Backward
            dx = self.afterCNN.backward(dz[i])

            dx1 = dx[:, :32, :, :]
            dx2 = dx[:, 32:, :, :]

            dx0 = self.CNN2Sequential.backward(dx2)
            self.CNNSequential.backward(dx0 + dx1)

        # RNN Backward
        self.LSTM.backward(dz)

    # inferences for Max length
    def evaluate(self, image, dataloader):
        index_list = list()
        image = image[0].reshape((1, 3, 64, 64))
        embed = dataloader.embed_Matrix[2].reshape((1, -1))
        h0 = np.zeros((1, 256))
        c0 = np.zeros((1, 256))
        for _ in range(dataloader.MAX_LENGTH):
            x1 = self.CNNSequential(image)
            x2 = self.CNN2Sequential(x1)
            x = np.concatenate((x1, x2), axis=1)
            x = self.afterCNN(x)

            # RNN Encoder
            y, (h0, c0) = self.LSTM.sequential[0].cell_forward(embed, (h0, c0))
            y = np.tanh(y)

            # ADD
            z = x + y
            # Last Layer of Networks
            output = self.lastLayer(z)
            pred_class = np.argmax(output, axis=-1).item(0)
            embed = dataloader.embed_Matrix[pred_class].reshape((1, -1))
            word = dataloader.token.index_word[pred_class]
            index_list.append(word)

            if pred_class is 3:
                break

        return index_list

    # save parameters
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    # load items
    def load(self, path):
        pickle.load(open(path, "rb"))


# for image captionning for RNN
class ImageCaptionRNN(object):
    def __init__(self, embed, dropout=False):
        super().__init__()
        self.CNNSequential = SeqModule([
            # convolution 1 + Relu
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            # convolution 2 + Relu + pooling
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # convolution 3 + Relu + pooling
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # convolution 4 + Relu + pooling
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        # 5층까지
        self.CNN2Sequential = SeqModule([
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
        ])
        ############
        # Concatenate
        ############
        self.afterCNN = SeqModule([
            nn.MaxPool2d(2),
            # same as reshape
            nn.flatten(),
            nn.Dropout(check=dropout),
            nn.linear(1536, 256),
            nn.tanh()
        ])

        self.Dropout = nn.Dropout(check=dropout)
        self.RNN = SeqModule([
            nn.RNN(embed, 256),
            nn.tanh(),
        ])
        ############
        # 중간에 ADD
        ############

        self.lastLayer = SeqModule([
            nn.linear(256, 256),
            nn.ReLU(),
            nn.linear(256, 1665)
        ])

    def __call__(self, images, embed):
        return self.forward(images, embed)

    # forward each layer
    def forward(self, images, embed):
        # CNN Encoder
        x1 = self.CNNSequential(images)
        x2 = self.CNN2Sequential(x1)
        x = np.concatenate((x1, x2), axis=1)
        x = self.afterCNN(x)

        # RNN Encoder
        y = self.Dropout(embed)
        y = self.RNN(y)
        seq_len, batch, _ = y.shape

        z = np.zeros(y.shape)
        output = np.zeros((seq_len, batch, 1665))

        for i in range(seq_len):
            # ADD
            z[i] = x + y[i]
            # Last Layer of Networks
            output[i] = self.lastLayer(z[i])
        return output

    # update parameters
    def update(self, grad_output):
        seq_len, batch, _ = grad_output.shape
        dz = np.zeros((seq_len, batch, 256))

        for i in reversed(range(seq_len)):
            dz[i] = self.lastLayer.backward(grad_output[i])

            # CNN Backward
            dx = self.afterCNN.backward(dz[i])

            dx1 = dx[:, :32, :, :]
            dx2 = dx[:, 32:, :, :]

            dx0 = self.CNN2Sequential.backward(dx2)
            self.CNNSequential.backward(dx0 + dx1)

        # RNN Backward
        self.RNN.backward(dz)

    def evaluate(self, image, dataloader):
        index_list = list()
        image = image[0].reshape((1, 3, 64, 64))
        embed = dataloader.embed_Matrix[2].reshape((1, -1))
        h0 = np.zeros((1, 256))
        for _ in range(dataloader.MAX_LENGTH):
            x1 = self.CNNSequential(image)
            x2 = self.CNN2Sequential(x1)
            x = np.concatenate((x1, x2), axis=1)
            x = self.afterCNN(x)

            # RNN Encoder
            y, h0 = self.RNN.sequential[0].cell_forward(embed, h0)
            y = np.tanh(y)

            # ADD
            z = x + y
            # Last Layer of Networks
            output = self.lastLayer(z)
            pred_class = np.argmax(output, axis=-1).item(0)
            embed = dataloader.embed_Matrix[pred_class].reshape((1, -1))
            word = dataloader.token.index_word[pred_class]
            index_list.append(word)

            if pred_class is 3:
                break

        return index_list

    # save parameters
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    # load items
    def load(self, path):
        pickle.load(open(path, "rb"))


# for the CNN Test
class CNNTest(object):
    def __init__(self):
        super().__init__()
        self.CNNSequential = SeqModule([
            # convolution 1 + Relu
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            # convolution 2 + Relu + pooling
            nn.Conv2d(4, 4, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # convolution 3 + Relu + pooling
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            # convolution 4 + Relu + pooling
            nn.Conv2d(8, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        # 5층까지
        self.CNN2Sequential = SeqModule([
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
        ])
        ############
        # Concatenate
        ############
        self.afterCNN = SeqModule([
            nn.MaxPool2d(2, padding=1),
            # same as reshape
            nn.flatten(),
        ])

        self.lastLayer = SeqModule([
            nn.linear(384, 128),
            nn.ReLU(),
            nn.linear(128, 10)
        ])

    def __call__(self, images):
        return self.forward(images)

    # forward each layer
    def forward(self, images):
        # CNN Encoder
        x1 = self.CNNSequential(images)
        x2 = self.CNN2Sequential(x1)
        x = np.concatenate((x1, x2), axis=1)
        x = self.afterCNN(x)
        x = self.lastLayer(x)
        return x

    # update parameters
    def update(self, grad_output):
        x = self.lastLayer.backward(grad_output)
        x = self.afterCNN.backward(x)
        x1 = x[:, :8, :]
        x2 = x[:, 8:, :]
        x0 = self.CNN2Sequential.backward(x2)
        self.CNNSequential.backward(x0 + x1)

    # save parameters
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))


# for Shakespeare
class RNNTest(object):
    def __init__(self, embed):
        super().__init__()

        self.RNN = SeqModule([nn.RNN(65, 256)])
        self.linear = SeqModule([
            nn.linear(256, 65),
        ])

    def __call__(self, embed):
        return self.forward(embed)

    # forward each layer
    def forward(self, embed):

        x = self.RNN(embed)
        seq_len, batch, _ = embed.shape
        z = np.zeros(embed.shape)

        for i in range(seq_len):
            # ADD

            # Last Layer of Networks
            z[i] = self.linear(x[i])
        return z

    # update parameters
    def update(self, grad_output):
        seq_len, batch, _ = grad_output.shape

        z = np.zeros((seq_len, batch, 256))
        for i in reversed(range(seq_len)):
            z[i] = self.linear.backward(grad_output[i])
        self.RNN.backward(z)

    def evaluate(self, embed, dataloader):
        batch_size = dataloader.batch_size
        h0 = np.zeros((batch_size, 256))
        index_list = list()
        for _ in range(100):
            one_hot = np.zeros((batch_size, 65))
            y, h0 = self.RNN.sequential[0].cell_forward(embed, h0)
            z = self.linear(y)
            arg = np.argmax(z, axis=1)
            one_hot[np.arange(batch_size), arg] = 1.0
            embed = one_hot
            word = dataloader.idx2char[arg[0]]
            index_list.append(word)

        return index_list

    # save parameters
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))


# for LSTM Shakespeare
class LSTMTest(object):
    def __init__(self, embed):
        super().__init__()

        self.RNN = SeqModule([nn.LSTM(65, 256)])
        self.linear = SeqModule([
            nn.linear(256, 65),
        ])

    def __call__(self, embed):
        return self.forward(embed)

    # forward each layer
    def forward(self, embed):

        x = self.RNN(embed)
        seq_len, batch, _ = embed.shape
        z = np.zeros(embed.shape)

        for i in range(seq_len):
            # ADD

            # Last Layer of Networks
            z[i] = self.linear(x[i])
        return z

    # update parameters
    def update(self, grad_output):
        seq_len, batch, _ = grad_output.shape

        z = np.zeros((seq_len, batch, 256))
        for i in reversed(range(seq_len)):
            z[i] = self.linear.backward(grad_output[i])
        self.RNN.backward(z)

    def evaluate(self, embed, dataloader):
        batch_size = dataloader.batch_size
        h0 = np.zeros((batch_size, 256))
        c0 = np.zeros((batch_size, 256))
        index_list = list()
        for _ in range(100):
            one_hot = np.zeros((batch_size, 65))
            y, (h0, c0) = self.RNN.sequential[0].cell_forward(embed, (h0, c0))
            z = self.linear(y)
            arg = np.argmax(z, axis=1)
            one_hot[np.arange(batch_size), arg] = 1.0
            embed = one_hot
            word = dataloader.idx2char[arg[0]]
            index_list.append(word)

        return index_list

    # save parameters
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
