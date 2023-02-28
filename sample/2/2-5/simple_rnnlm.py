import numpy as np

from common.time_layers import TimeAffine, TimeEmbedding, TimeRNN, TimeSoftmaxWithLoss


class SimpleRnnlm:
    """単純なRNNLM

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        layers (list): レイヤ
        loss_layer (TimeSoftmaxWithLoss): 損失関数
        rnn_layer (TimeRNN): RNNレイヤ
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数
            wordvec_size (int): 単語の分散表現の次元数
            hidden_size (int): 中間層のノード数
        """
        # 重みの初期化(Xavierの初期値)
        embed_weight = (np.random.randn(vocab_size, wordvec_size) / 100).astype("f")
        rnn_weight_in = (
            np.random.randn(wordvec_size, hidden_size) / np.sqrt(wordvec_size)
        ).astype("f")
        rnn_weight_hidden = (
            np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        ).astype("f")
        rnn_bias = np.zeros(hidden_size).astype("f")
        affine_weight = (
            np.random.randn(hidden_size, vocab_size) / np.sqrt(hidden_size)
        ).astype("f")
        affine_bias = np.zeros(vocab_size).astype("f")

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_weight),
            TimeRNN(rnn_weight_in, rnn_weight_hidden, rnn_bias, stateful=True),
            TimeAffine(affine_weight, affine_bias),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 全ての重みと勾配をリストにまとめる
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, inputs, labels):
        """順伝播

        Args:
            inputs (ndarray): 入力
            labels (ndarray): 正解ラベル

        Returns:
            float: 損失関数
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        loss = self.loss_layer.forward(inputs, labels)
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (int): 上流から伝わってきた勾配

        Returns:
            dout (ndarray): 下流に伝える勾配
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        """隠れ状態のリセット"""
        self.rnn_layer.reset_state()
