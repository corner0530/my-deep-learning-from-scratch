import numpy as np

from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 重みの初期化
        weight1 = np.random.normal(loc=0.0, scale=0.01, size=(input_size, hidden_size))
        bias1 = np.zeros(hidden_size)
        weight2 = np.random.normal(loc=0.0, scale=0.01, size=(hidden_size, output_size))
        bias2 = np.zeros(output_size)

        # レイヤの生成
        self.layers = [Affine(weight1, bias1), Sigmoid(), Affine(weight2, bias2)]
        self.loss_layer = SoftmaxWithLoss()

        # すべての重みと勾配をリストにまとめる
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params  # + でリストの結合
            self.grads += layer.grads

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def forward(self, inputs, labels):
        score = self.predict(inputs)
        loss = self.loss_layer.forward(score, labels)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
