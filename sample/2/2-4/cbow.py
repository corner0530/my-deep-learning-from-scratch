import numpy as np
from negative_sampling_layer import NegativeSamplingLoss

from common.layers import Embedding


class CBOW:
    """CBOWモデル

    Attributes:
        in_layers (list): 入力層のリスト
        ns_loss (NegativeSamplingLoss): 負例サンプリングによる学習を行うレイヤ
        params (list): パラメータ
        grads (list): 勾配
    """
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数
            hidden_size (int): 中間層のニューロン数
            window_size (int): コンテキストのサイズ
            corpus (list): 単語IDのリスト
        """
        # 重みの初期化
        weight_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype("f")
        weight_out = 0.01 * np.random.randn(vocab_size, hidden_size).astype("f")

        # レイヤの生成
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(weight_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(
            weight_out, corpus, power=0.75, sample_size=5
        )

        # 全ての重みと勾配を配列にまとめる
        layers = self.in_layers + [self.ns_loss]
        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = weight_in

    def forward(self, contexts, target):
        """順伝播

        Args:
            contexts (ndarray): コンテキスト
            target (ndarray): ターゲット

        Returns:
            float: 損失関数の値
        """
        # 各レイヤの出力を求めて平均を取り，その後NegativeSamplingLossレイヤに渡す
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (int, optional): 上流から伝わってきた勾配
        """
        # NegativeSamplingLossレイヤから勾配を受け取り，各レイヤに伝播させる
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
