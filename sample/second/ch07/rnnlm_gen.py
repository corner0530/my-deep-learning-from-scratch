import numpy as np

from common.functions import softmax
from sample.second.ch06.better_rnnlm import BetterRnnlm
from sample.second.ch06.rnnlm import Rnnlm


class RnnlmGen(Rnnlm):
    """RNN言語モデルによる文章生成"""

    def generate(self, start_id, skip_ids=None, sample_size=100):
        """文章の生成

        Args:
            start_id (int): 最初に与える単語ID
            skip_ids (list): サンプリングで除外する単語のID
            sample_size (int): サンプリングする単語の数
        """
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)  # 各単語のスコアを出力
            p = softmax(score.flatten())  # スコアを正規化

            sampled = np.random.choice(len(p), size=1, p=p)  # pでサンプリング
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids


class BetterRnnlmGen(BetterRnnlm):
    """改良版RNN言語モデルによる文章生成"""

    def generate(self, start_id, skip_ids=None, sample_size=100):
        """文章の生成

        Args:
            start_id (int): 最初に与える単語ID
            skip_ids (list): サンプリングで除外する単語のID
            sample_size (int): サンプリングする単語の数
        """
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)  # 各単語のスコアを出力
            p = softmax(score.flatten())  # スコアを正規化

            sampled = np.random.choice(len(p), size=1, p=p)  # pでサンプリング
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
