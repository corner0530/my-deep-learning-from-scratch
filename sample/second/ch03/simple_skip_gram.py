import numpy as np

from common.layers import MatMul, SoftmaxWithLoss


class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数
            hidden_size (int): 中間層のノード数
        """
        # 重みの初期化
        weight_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype("f")
        weight_out = 0.01 * np.random.randn(hidden_size, vocab_size).astype("f")

        # レイヤの生成
        self.in_layer = MatMul(weight_in)
        self.out_layer = MatMul(weight_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        # 全ての重みと勾配をリストにまとめる
        layers = [self.in_layer, self.out_layer]
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
            target (ndarray): 正解ラベル

        Returns:
            loss (float): 損失関数の値
        """
        hidden = self.in_layer.forward(target)
        score = self.out_layer.forward(hidden)
        loss1 = self.loss_layer1.forward(score, contexts[:, 0])
        loss2 = self.loss_layer2.forward(score, contexts[:, 1])
        loss = loss1 + loss2
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (int): 上流から伝わってきた勾配
        """
        dloss1 = self.loss_layer1.backward(dout)
        dloss2 = self.loss_layer2.backward(dout)
        dscore = dloss1 + dloss2
        dhidden = self.out_layer.backward(dscore)
        self.in_layer.backward(dhidden)
        return None
