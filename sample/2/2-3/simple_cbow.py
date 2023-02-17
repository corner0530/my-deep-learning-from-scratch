import sys

sys.path.append(".")
import numpy as np

from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    """単純なCBOWモデル

    Attributes:
        vocab_size (int): 語彙数
        hidden_size (int): 中間層のノード数
        weight_in (ndarray): 入力層の重み
        weight_out (ndarray): 出力層の重み
        in_layer0 (MatMul): 入力層のレイヤ
        in_layer1 (MatMul): 入力層のレイヤ
        out_layer (MatMul): 出力層のレイヤ
        loss_layer (SoftmaxWithLoss): 損失関数のレイヤ
        params (list): 重みパラメータ
        grads (list): 勾配
        word_vecs (ndarray): 単語の分散表現
    """

    def __init__(self, vocab_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数
            hidden_size (int): 中間層のノード数
        """
        # 重みの初期化(32ビットの浮動小数点数で初期化)
        weight_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype("f")
        weight_out = 0.01 * np.random.randn(hidden_size, vocab_size).astype("f")

        # レイヤの生成
        self.in_layer0 = MatMul(weight_in)  # コンテキストで使用する単語の数だけ作る
        self.in_layer1 = MatMul(weight_in)  #
        self.out_layer = MatMul(weight_out)
        self.loss_layer = SoftmaxWithLoss()

        # 全ての重みと勾配をリストにまとめる
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # メンバ変数に単語の分散表現を設定
        self.word_vecs = weight_in

    def forward(self, contexts, target):
        """順伝播

        Args:
            contexts (ndarray): コンテキスト
            target (ndarray): 正解ラベル

        Returns:
            loss (float): 損失関数の値
        """
        hidden_layer0 = self.in_layer0.forward(contexts[:, 0])
        hidden_layer1 = self.in_layer1.forward(contexts[:, 1])
        hidden_layer = (hidden_layer0 + hidden_layer1) * 0.5
        score = self.out_layer.forward(hidden_layer)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (int): 上流から伝わってくる勾配
        """
        dloss = self.loss_layer.backward(dout)
        doutl = self.out_layer.backward(dloss)
        doutl *= 0.5
        self.in_layer1.backward(doutl)
        self.in_layer0.backward(doutl)
        return None
