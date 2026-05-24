from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
        self.training = True

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

    def set_training(self, mode):
        self.training = mode


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # 使用更小的初始化范围来避免梯度爆炸
        if initialize_method == np.random.normal:
            # He initialization for ReLU networks
            self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = np.matmul(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        self.grads['W'] = np.matmul(self.input.T, grad) / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        d_input = np.matmul(grad, self.W.T)
        return d_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

def im2col(X, kernel_size, stride=1, padding=0):
    """
    Convert image to column matrix using as_strided (no Python loops).
    X: [batch, channels, H, W]
    Returns: cols [batch * H_out * W_out, channels * kernel_size * kernel_size]
    """
    N, C, H, W = X.shape
    K = kernel_size
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    if padding > 0:
        X = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    shape = (N, C, H_out, W_out, K, K)
    strides = (X.strides[0], X.strides[1],
               X.strides[2] * stride, X.strides[3] * stride,
               X.strides[2], X.strides[3])
    patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

    cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, C * K * K).copy()
    return cols, (N, C, H, W, H_out, W_out)


def col2im(cols, input_shape, kernel_size, stride=1, padding=0):
    """
    Reverse im2col: convert columns back to image gradients.
    cols: [N * H_out * W_out, C * K * K]
    Returns: imgs [N, C, H, W]
    """
    N, C, H, W, H_out, W_out = input_shape
    K = kernel_size

    cols_reshaped = cols.reshape(N, H_out, W_out, C, K, K).transpose(0, 3, 1, 2, 4, 5)

    if padding > 0:
        H_pad, W_pad = H + 2 * padding, W + 2 * padding
    else:
        H_pad, W_pad = H, W

    imgs = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)
    for i in range(K):
        for j in range(K):
            imgs[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride] += \
                cols_reshaped[:, :, :, :, i, j]

    if padding > 0:
        imgs = imgs[:, :, padding:-padding, padding:-padding]
    return imgs


class conv2D(Layer):
    """
    The 2D convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight: [out_channels, in_channels, kernel_size, kernel_size]
        # He initialization
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.input_shape = None
        self.cols = None  # cached im2col result from forward

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch, in_channels, H, W]
        out: [batch, out_channels, H_out, W_out]
        """
        self.input = X
        N, C, H, W = X.shape
        K = self.kernel_size

        # im2col: [N, C, H, W] -> [N * H_out * W_out, C * K * K]
        self.cols, self.input_shape = im2col(X, K, self.stride, self.padding)
        _, _, _, _, H_out, W_out = self.input_shape

        # W: [out_channels, in_channels, K, K] -> [out_channels, C * K * K]
        W_flat = self.W.reshape(self.out_channels, -1)

        # output: [N * H_out * W_out, out_channels]
        out = np.matmul(self.cols, W_flat.T) + self.b

        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grads):
        """
        grads: [batch, out_channels, H_out, W_out]
        return: [batch, in_channels, H, W]
        """
        N, C_out, H_out, W_out = grads.shape
        batch_size = N
        K = self.kernel_size

        # Reshape grads to [N * H_out * W_out, out_channels]
        grads_flat = grads.transpose(0, 2, 3, 1).reshape(-1, C_out)

        # W_flat: [out_channels, C * K * K]
        W_flat = self.W.reshape(self.out_channels, -1)

        # col: [N * H_out * W_out, C * K * K] (cached from forward)
        col = self.cols

        # grad_W: [out_channels, C * K * K] -> [out_channels, in_channels, K, K]
        self.grads['W'] = np.matmul(grads_flat.T, col) / batch_size
        self.grads['W'] = self.grads['W'].reshape(self.out_channels, self.in_channels, K, K)

        # grad_b: [out_channels] — normalize by spatial size to prevent bias explosion
        self.grads['b'] = np.sum(grads_flat, axis=0) / (batch_size * H_out * W_out)

        # grad_input via col2im
        grad_col = np.matmul(grads_flat, W_flat)  # [N * H_out * W_out, C * K * K]
        grad_input = col2im(grad_col, self.input_shape, K, self.stride, self.padding)

        return grad_input

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

    def set_training(self, mode):
        self.training = mode


class Dropout(Layer):
    """
    Inverted dropout: scales kept neurons by 1/(1-p) during training,
    does nothing during evaluation.
    """
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p
        self.mask = None
        self.optimizable = False
        self.training = True

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if self.training and self.p > 0:
            self.mask = np.random.binomial(1, 1 - self.p, size=X.shape).astype(X.dtype)
            self.mask /= (1 - self.p)
            return X * self.mask
        else:
            self.mask = None
            return X

    def backward(self, grads):
        if self.mask is not None:
            return grads * self.mask
        return grads

    def set_training(self, mode):
        self.training = mode


class MaxPool2D(Layer):
    """2D Max Pooling: [N, C, H, W] -> [N, C, H//S, W//S]"""
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_shape = None
        self.max_indices = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        N, C, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1

        cols, self.input_shape = im2col(X, K, S, 0)
        cols_reshaped = cols.reshape(N * H_out * W_out * C, K * K)
        self.max_indices = np.argmax(cols_reshaped, axis=1)
        max_vals = cols_reshaped[np.arange(len(self.max_indices)), self.max_indices]
        out = max_vals.reshape(N, H_out, W_out, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, grads):
        N, C, H_out, W_out = grads.shape
        K = self.kernel_size
        S = self.stride

        grads_flat = grads.transpose(0, 2, 3, 1).reshape(-1)

        mask = np.zeros((N * H_out * W_out * C, K * K), dtype=grads.dtype)
        mask[np.arange(len(self.max_indices)), self.max_indices] = grads_flat

        mask_cols = mask.reshape(N * H_out * W_out, C * K * K)
        grad_input = col2im(mask_cols, self.input_shape, K, S, 0)
        return grad_input


class Flatten(Layer):
    """Flatten a 4D tensor to 2D: [N, C, H, W] -> [N, C*H*W]"""
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.predicts = None
        self.labels = None
        self.grads = None
        self.has_softmax = True
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]

        # softmax with numerical stability
        x_max = np.max(predicts, axis=1, keepdims=True)
        x_exp = np.exp(predicts - x_max)
        softmax_output = x_exp / np.sum(x_exp, axis=1, keepdims=True)

        # compute cross-entropy loss
        one_hot_labels = np.eye(self.max_classes)[labels]
        loss = -np.sum(one_hot_labels * np.log(softmax_output + 1e-10)) / batch_size

        return loss

    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]

        # softmax output
        x_max = np.max(self.predicts, axis=1, keepdims=True)
        x_exp = np.exp(self.predicts - x_max)
        softmax_output = x_exp / np.sum(x_exp, axis=1, keepdims=True)

        # one-hot encoding
        one_hot_labels = np.eye(self.max_classes)[self.labels]

        # gradient = softmax - one_hot (this is dL/dlogits)
        self.grads = (softmax_output - one_hot_labels)

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition