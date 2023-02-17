# coding: utf-8
import numpy as np

from common.layers import Embedding


class EmbeddingDot:
    """Embeddingレイヤと内積の演算を組み合わせたレイヤ

    Attributes:
        embed (Embedding): Embeddingレイヤ
        params (list): パラメータ
        grads (list): 勾配
        cache (tuple): 順伝播時の中間データ
    """
    def __init__(self, weight):
        """コンストラクタ

        Args:
            weight (ndarray): 重み
        """
        self.embed = Embedding(weight)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, hidden, idx):
        """順伝播

        Args:
            hidden (ndarray): 中間層の出力
            idx (ndarray): 単語のID

        Returns:
            ndarray: 出力
        """
        target_weight = self.embed.forward(idx)
        out = np.sum(target_weight * hidden, axis=1)

        self.cache = (hidden, target_weight)
        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        hidden, target_weight = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_weight = dout * hidden
        self.embed.backward(dtarget_weight)
        dhidden = dout * target_weight
        return dhidden
