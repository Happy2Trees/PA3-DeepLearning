import numpy as np

import model.function as F


class baseLayer():
    def __init__(self):
        pass

    def __call__(self, input):
        output = self.forward(input)
        return output

    def forward(self, input):
        '''warping part'''
        pass

    def backward(self, input, output_grad):
        pass


class RNNBase():
    def __init__(self):
        pass

    def __call__(self, input, prev_state=None):
        output = self.forward(input, prev_state)
        return output

    def forward(self, input, prev_state=None):
        '''warping part'''
        pass

    def backward(self, input, output_grad):
        pass


class leakyReLU(baseLayer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.has_param = False

    def forward(self, x):
        return np.where(x > 0, x, x * self.alpha)

    # gradient will be 1 or alpha
    def backward(self, input, grad_output):
        grad = np.where(input > 0, 1, self.alpha)
        return grad * grad_output


class ReLU(baseLayer):
    def __init__(self):
        super().__init__()
        self.has_param = False

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, input, grad_output):
        grad = input > 0
        return grad * grad_output


# reference CS231n class of stanford university
# reference : https://cs231n.github.io/convolutional-networks
# reference : https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/

class Conv2d(baseLayer):
    def __init__(self, input_features, output_features, kernel_size, padding=1, stride=1):
        super().__init__()
        self.has_param = True
        self.input_features = input_features
        self.output_features = output_features
        self.stride = stride
        self.padding = padding

        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height = kernel_size[0]
            self.kernel_width = kernel_size[1]

        # kaiming he uniform distribution
        # gain = sqrt(5)
        # kaiming He initialization paper : https://arxiv.org/pdf/1502.01852.pdf
        self.weight = np.random.uniform(low=-1 / np.sqrt(input_features * self.kernel_height * self.kernel_width),
                                        high=1 / np.sqrt(input_features * self.kernel_height * self.kernel_width),
                                        size=(output_features, input_features, self.kernel_height, self.kernel_width))
        self.w_optim = F.optimizer()

        # initialize with kaiming he uniform distribution
        self.bias = np.random.uniform(low=-1 / np.sqrt(input_features * self.kernel_height * self.kernel_width),
                                      high=1 / np.sqrt(input_features * self.kernel_height * self.kernel_width),
                                      size=(output_features, 1))
        self.b_optim = F.optimizer()

    def forward(self, x):
        batch, _, input_height, input_width = x.shape
        # calculate output size because we have to calculate matrix form
        # to use
        self.out_height = (input_height + 2 * self.padding - self.kernel_height) // self.stride + 1
        self.out_width = (input_width + 2 * self.padding - self.kernel_width) // self.stride + 1
        # to calculate matrix form , using image to column function
        # weight_col ==> (output_features, input_features * kernel_height * kernel_width)
        weight_col = self.weight.reshape(self.output_features, -1)
        input_col = self.img_to_col(x)

        # forward
        out = weight_col @ input_col + self.bias

        # reshape the output to original form
        out = out.reshape(self.output_features, self.out_height, self.out_width, batch)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, input, output_grad):
        input_col = self.img_to_col(input)
        # excep output_features summing other axis
        grad_bias = np.sum(output_grad, axis=(0, 2, 3))
        grad_bias = grad_bias.reshape(self.output_features, -1)

        # output_grad reshaped ==> (output_Features,  output_height * output_width * batch)
        output_grad_reshaped = output_grad.transpose(1, 2, 3, 0).reshape(self.output_features, -1)
        grad_weight_reshaped = output_grad_reshaped @ input_col.T
        grad_weight = grad_weight_reshaped.reshape(self.weight.shape)

        # weight_reshaped --> (output_features, input_features * kernel_height * kernel_width)
        weight_reshaped = self.weight.reshape(self.output_features, -1)
        # grad_input_col ==> (input_Features * kernel_height * kernel_width, output_height * output_width * batch)
        grad_input_col = weight_reshaped.T @ output_grad_reshaped
        # multiply two matrix and return to original input matrix form
        grad_input = self.col_to_img(grad_input_col, input.shape)

        # gradien descent
        self.weight = self.weight - self.w_optim(grad_weight)
        self.bias = self.bias - self.b_optim(grad_bias)

        return grad_input

    def img_to_col(self, x):
        zero_x = np.array(x)
        if self.padding > 0:
            # x shapels consist of 4 channels
            zero_x = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)],
                            mode='constant')
        k, i, j = self.img_to_col_indices(x.shape)

        cols = zero_x[:, k, i, j]
        # change to (kernel_height * kernel_width * channel, output_height * output_width * batch )
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_height * self.kernel_width * x.shape[1], -1)
        return cols

    def img_to_col_indices(self, x_shape):
        batch, input_features, input_height, input_width = x_shape

        # i_column correspond to the kernel index
        i_column = np.repeat(np.arange(self.kernel_height), self.kernel_width)
        i_column = np.tile(i_column, self.input_features)
        # i_row correspond to the output index
        i_row = self.stride * np.repeat(np.arange(self.out_height), self.out_width)

        # j_column correspond to the kernel index
        j_column = np.tile(np.arange(self.kernel_width), self.kernel_height * input_features)
        j_row = self.stride * np.tile(np.arange(self.out_width), self.out_height)

        # j_row correspond to the output index
        i = i_column.reshape(-1, 1) + i_row.reshape(1, -1)
        j = j_column.reshape(-1, 1) + j_row.reshape(1, -1)

        # This takes into account input features. This is because we have to multiply it as much as input_features.
        k = np.repeat(np.arange(self.input_features), self.kernel_height * self.kernel_width).reshape(-1, 1)

        return k, i, j

    def col_to_img(self, column, x_shape):
        batch, input_features, input_height, input_width = x_shape
        input_height_padded, input_width_padded = input_height + 2 * self.padding, input_width + 2 * self.padding
        x_padded = np.zeros((batch, input_features, input_width_padded, input_height_padded))
        k, i, j = self.img_to_col_indices(x_shape)
        column_reshaped = column.reshape(input_features * self.kernel_height * self.kernel_width, -1, batch)
        column_reshaped = column_reshaped.transpose(2, 0, 1)

        # padded array with column indicies k, i, j np.add.at add not duplicately
        np.add.at(x_padded, (slice(None), k, i, j), column_reshaped)

        # padded array to original array
        if self.padding is not 0:
            return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x_padded


# reference : https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
class MaxPool2d(baseLayer):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.has_param = False
        if isinstance(kernel_size, int):
            self.kernel_height = kernel_size
            self.kernel_width = kernel_size
        else:
            self.kernel_height = kernel_size[0]
            self.kernel_width = kernel_size[1]
        if stride is not None:
            self.stride = stride
        else:
            self.stride = kernel_size
        self.padding = padding

    def forward(self, x):
        batch, input_features, input_height, input_width = x.shape
        # To prevent the channel from sticking in the column form
        x_reshaped = x.reshape(batch * input_features, 1, input_height, input_width)

        # calculate output size
        self.out_height = int((input_height + 2 * self.padding - self.kernel_height) / self.stride + 1)
        self.out_width = int((input_width + 2 * self.padding - self.kernel_width) / self.stride + 1)

        # output will be (kernel_height * kernel_width, output_height * output_width * batch * input_features )
        x_reshaped_cols = self.img_to_col(x_reshaped)
        # we have to save this form to get backpropagation
        self.shape_x_cols = x_reshaped_cols.shape

        # to find max value of each kernel(column)
        # we have to keep max Index for backpropagation
        self.max_index = np.argmax(x_reshaped_cols, axis=0)

        # Finally, we get all the max value at each column
        out_reshaped = x_reshaped_cols[self.max_index, range(self.max_index.size)]

        # reshaped to each form
        # out will be output_height * output_width * batch * input_features
        out = out_reshaped.reshape(self.out_height, self.out_width, batch, input_features)

        # Transpose to original form
        out = out.transpose(2, 3, 0, 1)
        return out

    def backward(self, input, grad_output):
        batch, input_features, input_height, input_width = input.shape
        # max pooling backward has role which upsamples grad_outputs so

        # (kernel_height * kernel_width, output_height * output_width * batch * input_features )
        # initialize input gradient
        ones_col = np.zeros(self.shape_x_cols)

        # grad output ==> (batch, output_features, output_height, output_width), then flattened
        # grad output_flattened ==> output_height * output_width * batch * input_features
        # caution output features == input features in max pooling
        grad_output_flattened = grad_output.transpose(2, 3, 0, 1).ravel()

        # ones_col ==> (kernel_height * kernel_width, output_height * output_width * batch * input_features )
        ones_col[self.max_index, range(self.max_index.size)] = grad_output_flattened

        # change to original upsampled Image
        reshaped_size = (batch * input_features, 1, input_height, input_width)
        grad_input_reshaped = self.col_to_img(ones_col, reshaped_size)

        # reshape to the input image shape
        grad_input = grad_input_reshaped.reshape(input.shape)

        return grad_input

    def img_to_col(self, x):
        zero_x = x
        if self.padding > 0:
            # x shapels consist of 4 channels
            zero_x = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)],
                            mode='constant')
        k, i, j = self.img_to_col_indices(x.shape)

        cols = zero_x[:, k, i, j]
        # change to (kernel_height * kernel_width * channel, output_height * output_width * batch )
        cols = cols.transpose(1, 2, 0).reshape(self.kernel_height * self.kernel_width * x.shape[1], -1)
        return cols

    def img_to_col_indices(self, x_shape):
        batch, input_features, input_height, input_width = x_shape

        # i_column correspond to the kernel index
        i_column = np.repeat(np.arange(self.kernel_height), self.kernel_width)
        i_column = np.tile(i_column, input_features)
        # i_row correspond to the output index
        i_row = self.stride * np.repeat(np.arange(self.out_height), self.out_width)

        # j_column correspond to the kernel index
        j_column = np.tile(np.arange(self.kernel_width), self.kernel_height * input_features)
        j_row = self.stride * np.tile(np.arange(self.out_width), self.out_height)

        # j_row correspond to the output index numpy magic function
        i = i_column.reshape(-1, 1) + i_row.reshape(1, -1)
        j = j_column.reshape(-1, 1) + j_row.reshape(1, -1)

        # This takes into account input features. This is because we have to multiply it as much as input_features.
        k = np.repeat(np.arange(input_features), self.kernel_height * self.kernel_width).reshape(-1, 1)

        return k, i, j

    def col_to_img(self, column, x_shape):
        batch, input_features, input_height, input_width = x_shape
        input_height_padded, input_width_padded = input_height + 2 * self.padding, input_width + 2 * self.padding
        x_padded = np.zeros((batch, input_features, input_width_padded, input_height_padded))
        k, i, j = self.img_to_col_indices(x_shape)
        column_reshaped = column.reshape(input_features * self.kernel_height * self.kernel_width, -1, batch)
        column_reshaped = column_reshaped.transpose(2, 0, 1)

        # padded array with column indicies k, i, j np.add.at add not duplicately
        np.add.at(x_padded, (slice(None), k, i, j), column_reshaped)

        # padded array to original array
        if self.padding is not 0:
            return x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return x_padded


class Dropout(baseLayer):
    def __init__(self, check=False, p=0.5):
        super().__init__()
        self.has_param = False
        self.check = check
        if not check:
            self.p = 1
        else:
            self.p = p
        self.mask = None

    def forward(self, input):
        input = input * self.p
        self.mask = np.random.binomial(1.0, self.p, size=input.shape) / self.p
        return self.mask * input

    def backward(self, input, output_grad):
        if output_grad is None:
            return None
        return self.mask * output_grad


class linear(baseLayer):

    def __init__(self, input, output):
        super().__init__()
        self.has_param = True

        # kaiming he initialize
        k = 1 / np.sqrt(input)
        self.weight = np.random.normal(loc=0.0,
                                        scale = np.sqrt(2/(input+output)),
                                        size = (input,output))
        self.w_optim = F.optimizer()

        self.bias = np.zeros(output)
        self.b_optim = F.optimizer()

    def forward(self, x):
        return np.dot(x, self.weight) + self.bias

    def backward(self, input, grad_output):
        # for the previous layer, gradient update
        grad_input = np.dot(grad_output, self.weight.T)

        # we sum all of batches
        grad_weights = np.dot(input.T, grad_output)
        grad_bias = np.mean(grad_output, axis=0) * input.shape[0]

        # gradient descent step
        self.weight = self.weight - self.w_optim(grad_weights)
        self.bias = self.bias - self.b_optim(grad_bias)

        return grad_input


class flatten(baseLayer):
    def __init__(self):
        super().__init__()
        self.has_param = False

    def forward(self, input):
        batch, input_features, input_height, input_width = input.shape
        # reshape
        output = input.reshape(batch, -1)
        return output

    def backward(self, input, output_grad):
        batch, input_features, input_height, input_width = input.shape
        # output grad shape ==> (batch, input_features * input_height * input_width)
        grad_input = output_grad.reshape(batch, input_features, input_height, input_width)
        return grad_input


class tanh(baseLayer):
    def __init__(self):
        super().__init__()
        self.has_param = False

    def forward(self, input):
        values = np.tanh(input)
        return values

    def backward(self, input, output_grad):
        grad_input = F.derivative_tanh(input) * output_grad
        return grad_input


'''
:parameter
input_size –- The number of expected features in the input x
hidden_size -- The number of features in the hidden state h
bias -- If False, then the layer does not use bias weights
'''


class LSTMCell(baseLayer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.has_param = True
        # initialize with uniform distribution with sqrt(1/hiddensize)
        # weights for LSTM Cells
        # reference : https://pytorch.org/docs/stable/nn.html?highlight=nn%20lstm#torch.nn.LSTMCell

        # f, i, o, g --> 0, 1, 2, 3

        sqrt_k = 1 / np.sqrt(hidden_size)
        self.weight_ih = np.random.uniform(low=-6 / np.sqrt(4*hidden_size + input_size), high=6 / np.sqrt(4*hidden_size + input_size), size=(4 * hidden_size, input_size))
        self.w_ih_optim = F.optimizer()
        self.weight_hh = np.random.uniform(low=-6 / np.sqrt(4*hidden_size + hidden_size), high=-6 / np.sqrt(4*hidden_size + hidden_size), size=(4 * hidden_size, hidden_size))
        self.w_hh_optim = F.optimizer()

        self.bias_ih = np.random.uniform(low=-1 / sqrt_k, high=1 / sqrt_k, size=(4 * hidden_size))
        self.b_ih_optim = F.optimizer()

        self.input_size = input_size
        self.hidden_size = hidden_size

    # h_0 = tensor containing the initial hidden state for each element in the batch.
    # c_0 = tensor containing the initial cell state for each element in the batch.
    # input_size = (h_0, c_0) => (batch, hidden_size)
    # output size = (h_1, c_1) => (batch, hidden_size)
    # inpu = (batch, )
    def forward(self, input):
        inpu, (h0, c0) = input

        # preact --> (batch, 4*hidden_size)
        # magic function of numpy bias will be upgraded to bias
        preact = (inpu @ self.weight_ih.T + self.bias_ih) + h0 @ self.weight_hh.T

        gates = F.sigmoid(preact[:, :3 * self.hidden_size])
        g_t = np.tanh(preact[:, 3 * self.hidden_size:])
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, 2 * self.hidden_size:3 * self.hidden_size]

        # f, i, g, o --> (batch_size, hidden_size)
        c_t = c0 * f_t + i_t * g_t

        h_t = o_t * np.tanh(c_t)

        # (batch, i|f|o|g|c)
        states = {'preact': preact, 'Gate': np.column_stack((gates, g_t)), 'ct': c_t}

        return h_t, (h_t, c_t), states

    # backward update and parameters
    # output_grad --> (batch, hidden_size)
    # it is initialized with h_t
    def backward(self, input, output_grad):
        # for back propagation
        inpu, (h0, c0), states = input
        grad_output, (dh_next, dc_next) = output_grad
        grad_output = grad_output + dh_next

        preact = states['preact']
        i_b = preact[:, :self.hidden_size]
        f_b = preact[:, self.hidden_size:2 * self.hidden_size]
        o_b = preact[:, 2 * self.hidden_size:3 * self.hidden_size]
        g_b = preact[:, 3 * self.hidden_size:4 * self.hidden_size]

        gates = states['Gate']
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, 2 * self.hidden_size:3 * self.hidden_size]
        g_t = gates[:, 3 * self.hidden_size:4 * self.hidden_size]

        c_t = states['ct']

        # we do not implement y because of 유연성

        grad_ct = F.derivative_tanh(c_t) * (o_t * grad_output) + dc_next

        grad_it = F.derivative_sigmoid(i_b) * g_t * grad_ct
        grad_ft = F.derivative_sigmoid(f_b) * c0 * grad_ct
        grad_ot = F.derivative_sigmoid(o_b) * np.tanh(c_t) * grad_output
        grad_gt = F.derivative_tanh(g_b) * i_t * grad_ct

        # preactivation grad
        grad_preact = np.column_stack((grad_it, grad_ft, grad_ot, grad_gt))

        # calculate for previous layer gradient
        dh_before = grad_preact @ self.weight_hh
        dc_before = f_t * grad_ct

        grad_xh = grad_preact.T @ inpu
        self.weight_ih = self.weight_ih - self.w_ih_optim(grad_xh)
        # bias must be reduced batch
        self.bias_ih = self.bias_ih - self.b_ih_optim(np.mean(grad_preact, axis=0) * inpu.shape[0])
#        np.clip(self.bias_ih, -5, 5, out=self.bias_ih)
#        np.clip(self.weight_ih, -1, 1, out=self.weight_ih)

        grad_hh = grad_preact.T @ h0
        self.weight_hh = self.weight_hh - self.w_hh_optim(grad_hh)
#        np.clip(self.weight_hh, -1, 1, out=self.weight_hh)

        return dh_before, dc_before


# for the Base wrapped
class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.has_param = True
        # initialize with uniform distribution with sqrt(1/hiddensize)
        # weights for LSTM Cells
        # reference : https://pytorch.org/docs/stable/nn.html?highlight=nn%20lstm#torch.nn.LSTMCell

        self.Cell = LSTMCell(input_size, hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = list()

    # h_0 = tensor containing the initial hidden state for each element in the batch.
    # c_0 = tensor containing the initial cell state for each element in the batch.
    # input_size = (h_0, c_0) => (batch, hidden_size)
    # output size = (h_1, c_1) => (batch, hidden_size)

    def forward(self, input, prev_state=None):
        seqlen, batch, input_size = input.shape
        if prev_state is None:
            h0 = np.zeros((batch, self.hidden_size))
            c0 = np.zeros((batch, self.hidden_size))
        else:
            h0, c0 = prev_state

        output = np.zeros((seqlen, batch, self.hidden_size))
        self.activation = list()
        for index in range(seqlen):
            self.activation.append([input[index], (h0, c0)])
            hidden_output, (h0, c0), states = self.Cell.forward((input[index], (h0, c0)))
            self.activation[index].append(states)
            output[index] = hidden_output

        return output

    # backward update and parameters
    # output_grad --> (seq_len, batch, hidden_size)
    # it is initialized with h_t
    def backward(self, input, output_grad):
        # for back propagation
        seqlen, batch, input_size = input.shape
        dh_next = np.zeros((batch, self.hidden_size))
        dc_next = np.zeros((batch, self.hidden_size))

        for index in reversed(range(seqlen)):
            dh_next, dc_next = self.Cell.backward(self.activation[index], (output_grad[index], (dh_next, dc_next)))

    def cell_forward(self, input, state):
        h0, c0 = state
        hidden_output, (h0, c0), states = self.Cell.forward((input, (h0, c0)))
        return hidden_output, (h0, c0)


# cell
class RNNCell(baseLayer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.has_param = True
        # initialize with uniform distribution with sqrt(1/hiddensize)
        # reference : https://pytorch.org/docs/stable/nn.html?highlight=nn%20lstm#torch.nn.LSTMCell

        # f, i, o, g --> 0, 1, 2, 3

        sqrt_k = np.sqrt(hidden_size)
        self.weight_ih = np.random.uniform(low=-6 / np.sqrt(hidden_size + input_size), high=6 / np.sqrt(hidden_size + input_size), size=(hidden_size, input_size))
        self.w_ih_optim = F.optimizer()
        self.weight_hh = np.random.uniform(low=-6 / np.sqrt(hidden_size + hidden_size), high=-6 / np.sqrt(hidden_size + hidden_size), size=(hidden_size, hidden_size))
        self.w_hh_optim = F.optimizer()

        self.bias_ih = np.random.uniform(low=-1 / sqrt_k, high=1 / sqrt_k, size=(hidden_size))
        self.b_ih_optim = F.optimizer()

        self.input_size = input_size
        self.hidden_size = hidden_size

    # h_0 = tensor containing the initial hidden state for each element in the batch.
    # c_0 = tensor containing the initial cell state for each element in the batch.
    # input_size = (h_0, c_0) => (batch, hidden_size)
    # output size = (h_1, c_1) => (batch, hidden_size)
    # inpu = (batch, embedding matrix size)
    def forward(self, input):
        inpu, h0 = input

        # preact --> (batch, 4*hidden_size)
        # magic function of numpy bias will be upgraded to bias
        preact = inpu @ self.weight_ih.T + self.bias_ih + h0 @ self.weight_hh.T

        h_t = np.tanh(preact)
        states = {'h_b': preact}
        return h_t, h_t, states

    # backward update and parameters
    # output_grad --> (batch, hidden_size)
    # it is initialized with h_t
    def backward(self, input, output_grad):
        # for back propagation
        inpu, h0, states = input
        grad_output, dh_next = output_grad
        grad_output = grad_output + dh_next

        h_b = states['h_b']

        grad_ht = F.derivative_tanh(h_b) * grad_output

        # calculate for previous layer gradient
        dh_before = grad_ht @ self.weight_hh

        grad_xh = grad_ht.T @ inpu
        self.weight_ih = self.weight_ih - self.w_ih_optim(grad_xh)
        self.bias_ih = self.bias_ih - self.b_ih_optim(np.mean(grad_ht, axis=0) * inpu.shape[0])

#        np.clip(self.bias_ih, -5, 5, out=self.bias_ih)
#        np.clip(self.weight_ih, -5, 5, out=self.weight_ih)


        grad_hh = grad_ht.T @ h0
        self.weight_hh = self.weight_hh - self.w_hh_optim(grad_hh)
#        np.clip(self.weight_hh, -5, 5, out=self.weight_hh)


        return dh_before


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.has_param = True
        # initialize with uniform distribution with sqrt(1/hiddensize)
        # weights for LSTM Cells
        # reference : https://pytorch.org/docs/stable/nn.html?highlight=nn%20lstm#torch.nn.LSTMCell

        self.Cell = RNNCell(input_size, hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = list()

    # h_0 = tensor containing the initial hidden state for each element in the batch.
    # c_0 = tensor containing the initial cell state for each element in the batch.
    # input_size = (h_0, c_0) => (batch, hidden_size)
    # output size = (h_1, c_1) => (batch, hidden_size)

    def forward(self, input, prev_state=None):
        seqlen, batch, input_size = input.shape
        if prev_state is None:
            h0 = np.zeros((batch, self.hidden_size))
        else:
            h0 = prev_state

        output = np.zeros((seqlen, batch, self.hidden_size))
        self.activation = list()
        for index in range(seqlen):
            self.activation.append([input[index], h0])
            hidden_output, h0, states = self.Cell.forward((input[index], h0))
            self.activation[index].append(states)
            output[index] = hidden_output
        h_t = h0

        return output

    # backward update and parameters
    # output_grad --> (seq_len, batch, hidden_size)
    # it is initialized with h_t
    def backward(self, input, output_grad):
        # for back propagation
        seqlen, batch, input_size = input.shape
        dh_next = np.zeros((batch, self.hidden_size))
        #        not implemented yet because there are no more previous structures in our works
        #        grad_input = np.zeros((seqlen, batch, input_size))

        for index in reversed(range(seqlen)):
            dh_next = self.Cell.backward(self.activation[index], (output_grad[index], dh_next))

    def cell_forward(self, input, state):
        h0 = state
        hidden_output, h0, states = self.Cell.forward((input, h0))
        return hidden_output, h0
