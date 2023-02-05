# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
from collections import OrderedDict

import numpy as np

from common.layers import *


class SimpleConvNet:
    def __init__(
        self,
        input_dim=(1, 28, 28),
        hidden_size=100,
        output_size=10,
        weight_init_std=0.01,
    ):
        filter_num = 30
        filter_size = 5
        filter_pad = 0
        filter_stride = 1
        input_size = input_dim[1]
        conv_output_size = (
            input_size - filter_size + 2 * filter_pad
        ) / filter_stride + 1
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        # 重みの初期化
        weight1 = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        bias1 = np.zeros(filter_num)
        weight2 = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        bias2 = np.zeros(hidden_size)
        weight3 = weight_init_std * np.random.randn(hidden_size, output_size)
        bias3 = np.zeros(output_size)

        # レイヤの生成
        self.layers = [
            Convolution(weight1, bias1, filter_stride, filter_pad),
            Relu(),
            Pooling(pool_height=2, pool_width=2, stride=2),
            Affine(weight2, bias2),
            Relu(),
            Affine(weight3, bias3),
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]
