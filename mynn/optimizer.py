from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    # 检查梯度是否为None，避免TypeError
                    if layer.grads[key] is None:
                        continue
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
        for i, layer in enumerate(model.layers):
            if layer.optimizable:
                self.velocity[i] = {}
                for key in layer.params.keys():
                    self.velocity[i][key] = np.zeros_like(layer.params[key])

    def step(self):
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    # 检查梯度是否为None，避免TypeError
                    if layer.grads[key] is None:
                        continue
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    # update velocity
                    self.velocity[i][key] = self.mu * self.velocity[i][key] + layer.grads[key]
                    # update parameters
                    layer.params[key] -= self.init_lr * self.velocity[i][key]