import collections

from common.layers import Embedding, SigmoidWithLoss
from common.np import GPU, np  # import numpy as np


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
        target_weight = self.embed.forward(idx)  # Embeddingレイヤの順伝播
        out = np.sum(target_weight * hidden, axis=1)  # 内積の計算

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


class UnigramSampler:
    """負例をサンプリングするクラス

    Attributes:
        sample_size (int): 負例のサンプル数
        vocab_size (int): 語彙数
        word_p (ndarray): 各単語の確率
    """

    def __init__(self, corpus, power, sample_size):
        """コンストラクタ

        Args:
            corpus (list): 単語IDのリスト
            power (float): 累乗のpower
            sample_size (int): 負例のサンプル数
        """
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # 単語の出現回数をカウント
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        # 語彙数を計算
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        # 単語の出現回数を並べたリストを作成
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        # 単語の出現回数をpower乗して総和で割ったものを確率とする
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        """指定したものを正例として，それ以外の単語のサンプリングを行う

        Args:
            target (ndarray): 正例の単語ID

        Returns:
            ndarray: 負例の単語ID
        """
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(
                    self.vocab_size, size=self.sample_size, replace=False, p=p
                )
        else:
            negative_sample = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.sample_size),
                replace=True,
                p=self.word_p,
            )

        return negative_sample


class NegativeSamplingLoss:
    """Negative Samplingを行い損失関数を計算するレイヤ

    Attributes:
        loss_layers (list): SigmoidWithLossレイヤのリスト
        embed_dot_layers (list): EmbeddingDotレイヤのリスト
        params (list): パラメータ
        grads (list): 勾配
        sample_size (int): 負例のサンプル数
        sampler (UnigramSampler): 負例をサンプリングするクラス
    """

    def __init__(self, weight, corpus, power=0.75, sample_size=5):
        """コンストラクタ

        Args:
            weight (ndarray): 出力側の重み
            corpus (list): 単語IDのリスト
            power (float, optional): 累乗の指数
            sample_size (int, optional): 負例のサンプル数
        """
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [
            SigmoidWithLoss() for _ in range(sample_size + 1)
        ]  # 正例1個(最初のレイヤ) + 負例sample_size個
        self.embed_dot_layers = [EmbeddingDot(weight) for _ in range(sample_size + 1)]

        # パラメータと勾配をリストにまとめる
        self.params = []
        self.grads = []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, hidden, target):
        """順伝播

        Args:
            hidden (ndarray): 中間層の出力
            target (ndarray): 正例の単語ID

        Returns:
            float: 損失関数の値
        """
        batch_size = target.shape[0]

        # 負例のサンプリング
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例についての順伝播
        score = self.embed_dot_layers[0].forward(hidden, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 負例についての順伝播
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(hidden, negative_target)
            loss = self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dhidden = 0
        for layer0, layer1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = layer0.backward(dout)
            dhidden += layer1.backward(dscore)

        return dhidden
