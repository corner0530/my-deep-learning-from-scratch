"""最適化手法の実装

Attributes:
    SGD (class): 確率的勾配降下法
    Momentum (class): Momentum SGD
    Nesterov (class): Nesterov's Accelerated Gradient
    AdaGrad (class): AdaGrad
    RMSprop (class): RMSprop
    AdaDelta (class): AdaDelta
    Adam (class): Adam
"""
from common.np import np  # import numpy as np


class SGD:
    """確率的勾配降下法

    Attributes:
        lr (float): 学習率
    """

    def __init__(self, lr=0.01):
        """コンストラクタ

        Args:
            lr (float): 学習率
        """
        self.lr = lr

    def update(self, params, grads):
        """パラメータの更新

        Args:
            params (list): パラメータ
            grads (list): 勾配
        """
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Momentum:
    """Momentum SGD

    Attributes:
        lr (float): 学習率
        momentum (float): 慣性項のスケール
        v (list): 速度
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """コンストラクタ

        Args:
            lr (float): 学習率
            momentum (float): 運動量
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """パラメータの更新

        Args:
            params (list): パラメータ
            grads (list): 勾配
        """
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


class Nesterov:
    """Nesterov's Accelerated Gradient

    Attributes:
        lr (float): 学習率
        momentum (float): 慣性項のスケール
        v (list): 速度
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """コンストラクタ

        Args:
            lr (float): 学習率
            momentum (float): 運動量
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """パラメータの更新

        Args:
            params (list): パラメータ
            grads (list): 勾配
        """
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += (
                self.momentum * self.momentum * self.v[i]
                - (1 + self.momentum) * self.lr * grads[i]
            )


class AdaGrad:
    """AdaGrad

    Attributes:
        lr (float): 学習率
        h (list): 勾配の二乗和
    """

    def __init__(self, lr=0.01):
        """コンストラクタ

        Args:
            lr (float): 学習率
        """
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        """パラメータの更新

        Args:
            params (list): パラメータ
            grads (list): 勾配
        """
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class RMSprop:
    """RMSprop

    Attributes:
        lr (float): 学習率
        decay_rate (float): 減衰率
        h (list): 勾配の二乗和
    """

    def __init__(self, lr=0.01, decay_rate=0.99):
        """コンストラクタ

        Args:
            lr (float): 学習率
            decay_rate (float): 減衰率
        """
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        """パラメータの更新

        Args:
            params (list): パラメータ
            grads (list): 勾配
        """
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] = (
                self.decay_rate * self.h[i]
                + (1 - self.decay_rate) * grads[i] * grads[i]
            )
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class AdaDelta:
    """AdaDelta

    Attributes:
        decay_rate (float): 減衰率
        eps (float): 小さな値
        h (list): 勾配の二乗和
        delta (list): 更新量の二乗和
    """

    def __init__(self, decay_rate=0.95, eps=1e-6):
        """コンストラクタ

        Args:
            decay_rate (float): 減衰率
            eps (float): 小さな値
        """
        self.decay_rate = decay_rate
        self.eps = eps
        self.h = None
        self.delta = None

    def update(self, params, grads):
        """パラメータの更新

        Args:
            params (list): パラメータ
            grads (list): 勾配
        """
        if self.h is None:
            self.h = []
            self.delta = []
            for param in params:
                self.h.append(np.zeros_like(param))
                self.delta.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] = (
                self.decay_rate * self.h[i]
                + (1 - self.decay_rate) * grads[i] * grads[i]
            )
            delta = (
                -grads[i]
                * np.sqrt(self.delta[i] + self.eps)
                / np.sqrt(self.h[i] + self.eps)
            )
            self.delta[i] = (
                self.decay_rate * self.delta[i] + (1 - self.decay_rate) * delta * delta
            )
            params[i] += delta


class Adam:
    """Adam

    Attributes:
        lr (float): 学習率
        beta1 (float): 1次モーメントの減衰率
        beta2 (float): 2次モーメントの減衰率
        iter (int): 更新回数
        m (list): 1次モーメント
        v (list): 2次モーメント
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """コンストラクタ

        Args:
            lr (float): 学習率
            beta1 (float): 1次モーメントの減衰率
            beta2 (float): 2次モーメントの減衰率
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        """パラメータの更新

        Args:
            params (list): パラメータ
            grads (list): 勾配
        """
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i]
            self.v[i] = (
                self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i]
            )
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
