from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout=0.0):
        self.size_list = size_list
        self.act_func = act_func
        self.dropout = dropout

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if dropout > 0:
                        self.layers.append(Dropout(p=dropout))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def set_training(self, mode):
        for layer in self.layers:
            layer.set_training(mode)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    CNN for MNIST.
    architecture: conv->relu->pool->[dropout]->conv->relu->pool->[dropout]->flatten->fc1->relu->[dropout]->fc2
    """
    def __init__(self, conv_params=None, dropout=0.0):
        if conv_params is None:
            conv_params = [
                {'in_c': 1, 'out_c': 32, 'kernel_size': 3, 'padding': 1},
                {'in_c': 32, 'out_c': 64, 'kernel_size': 3, 'padding': 1},
            ]

        self.conv_params = conv_params
        self.dropout = dropout
        self.layers = []

        for cp in conv_params:
            p = cp.get('padding', 0)
            self.layers.append(conv2D(cp['in_c'], cp['out_c'], cp['kernel_size'], padding=p))
            self.layers.append(ReLU())
            self.layers.append(MaxPool2D(kernel_size=2, stride=2))
            if dropout > 0:
                self.layers.append(Dropout(p=dropout))

        self.layers.append(Flatten())

        H, W = 28, 28
        for cp in conv_params:
            k = cp['kernel_size']
            p = cp.get('padding', 0)
            H = (H + 2 * p - k) + 1
            W = (W + 2 * p - k) + 1
            H //= 2
            W //= 2
        flat_dim = conv_params[-1]['out_c'] * H * W

        self.layers.append(Linear(flat_dim, 128))
        self.layers.append(ReLU())
        if dropout > 0:
            self.layers.append(Dropout(p=dropout))
        self.layers.append(Linear(128, 10))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def set_training(self, mode):
        for layer in self.layers:
            layer.set_training(mode)

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.conv_params = param_list[0]
        self.layers = []
        n_conv = len(self.conv_params)
        for i, cp in enumerate(self.conv_params):
            p = cp.get('padding', 0)
            layer = conv2D(cp['in_c'], cp['out_c'], cp['kernel_size'], padding=p)
            layer.W = param_list[1 + i]['W']
            layer.b = param_list[1 + i]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            self.layers.append(layer)
            self.layers.append(ReLU())
            self.layers.append(MaxPool2D(kernel_size=2, stride=2))
            if self.dropout > 0:
                self.layers.append(Dropout(p=self.dropout))
        self.layers.append(Flatten())
        H, W = 28, 28
        for cp in self.conv_params:
            k = cp['kernel_size']
            p = cp.get('padding', 0)
            H = (H + 2 * p - k) + 1
            W = (W + 2 * p - k) + 1
            H //= 2
            W //= 2
        flat_dim = self.conv_params[-1]['out_c'] * H * W
        fc1 = Linear(flat_dim, 128)
        fc1.W = param_list[1 + n_conv]['W']
        fc1.b = param_list[1 + n_conv]['b']
        fc1.params['W'] = fc1.W
        fc1.params['b'] = fc1.b
        self.layers.append(fc1)
        self.layers.append(ReLU())
        if self.dropout > 0:
            self.layers.append(Dropout(p=self.dropout))
        fc2 = Linear(128, 10)
        fc2.W = param_list[2 + n_conv]['W']
        fc2.b = param_list[2 + n_conv]['b']
        fc2.params['W'] = fc2.W
        fc2.params['b'] = fc2.b
        self.layers.append(fc2)

    def save_model(self, save_path):
        param_list = [self.conv_params]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b']})
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)