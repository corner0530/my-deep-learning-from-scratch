from common.layers import Softmax
from common.np import np


class WeightSum:
    """重み付き和レイヤ

    Attributes:
        params (list): パラメータを保持するリスト
        grads (list): 勾配を保持するリスト
        cache (tuple): 順伝播時の中間データを保持するタプル
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.cache = None

    def forward(self, hiddens, weight):
        """順伝播

        Args:
            hiddens (ndarray): 隠れ状態
            weight (ndarray): 重み

        Returns:
            ndarray: 重み付き和
        """
        batch_num, times, hidden_size = hiddens.shape

        weight_repeat = weight.reshape(batch_num, times, 1).repeat(
            hidden_size, axis=2
        )  # 各単語のベクトルを重み付けするためコピーして変形
        tmp = hiddens * weight_repeat
        out = np.sum(tmp, axis=1)  # 各単語のベクトルについて重み付けして和を取りコンテキストベクトルを作成

        self.cache = (hiddens, weight_repeat)
        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 隠れ状態の勾配,
            ndarray: 重みの勾配
        """
        hiddens, weight_repeat = self.cache
        batch_num, times, hidden_size = hiddens.shape

        dt = dout.reshape(batch_num, 1, hidden_size).repeat(times, axis=1)  # sumの逆伝播
        dweight_repeat = dt * hiddens
        dhiddens = dt * weight_repeat
        dweight = np.sum(dweight_repeat, axis=2)  # repeatの逆伝播

        return dhiddens, dweight


class AttentionWeight:
    """アテンション重みレイヤ

    Attributes:
        params (list): パラメータを保持するリスト
        grads (list): 勾配を保持するリスト
        softmax (Softmax): Softmaxレイヤ
        cache (tuple): 順伝播時の中間データを保持するタプル
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.softmax = Softmax()
        self.cache = []

    def forward(self, hiddens, word_hidden):
        """順伝播

        Args:
            hiddens (ndarray): 隠れ状態
            word_hidden (ndarray): 単語の隠れ状態

        Returns:
            ndarray: 各単語の重み
        """
        batch_num, times, hidden_size = hiddens.shape

        word_hidden_repeat = word_hidden.reshape(batch_num, 1, hidden_size).repeat(
            times, axis=1
        )
        t = hiddens * word_hidden_repeat
        t_sum = np.sum(t, axis=2)  # 各単語との類似度(内積)を計算
        out = self.softmax.forward(t_sum)

        self.cache = (hiddens, word_hidden_repeat)
        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 隠れ状態の勾配,
            ndarray: 単語の隠れ状態の勾配
        """
        hiddens, word_hidden_repeat = self.cache
        batch_num, times, hidden_size = hiddens.shape

        dt_sum = self.softmax.backward(dout)
        dt = dt_sum.reshape(batch_num, times, 1).repeat(hidden_size, axis=2)
        dhiddens = dt * word_hidden_repeat
        dword_hidden_repeat = dt * hiddens
        dword_hidden = np.sum(dword_hidden_repeat, axis=1)

        return dhiddens, dword_hidden


class Attention:
    """Attentionレイヤ

    Attributes:
        params (list): パラメータを保持するリスト
        grads (list): 勾配を保持するリスト
        attention_weight_layer (AttentionWeight): AttentionWeightレイヤ
        weight_sum_layer (WeightSum): WeightSumレイヤ
        attention_weight (ndarray): 各単語の重み
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hiddens, word_hidden):
        """順伝播

        Args:
            hiddens (ndarray): Encoderの隠れ状態
            word_hidden (ndarray): 単語の隠れ状態

        Returns:
            ndarray: コンテキストベクトル
        """
        attention_weight = self.attention_weight_layer.forward(hiddens, word_hidden)
        out = self.weight_sum_layer.forward(hiddens, attention_weight)
        self.attention_weight = attention_weight
        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: Encoderの隠れ状態の勾配,
            ndarray: 単語の隠れ状態の勾配
        """
        dhiddens0, dattention_weight = self.weight_sum_layer.backward(dout)
        dhiddens1, dword_hidden = self.attention_weight_layer.backward(
            dattention_weight
        )
        dhiddens = dhiddens0 + dhiddens1
        return dhiddens, dword_hidden


class TimeAttention:
    """TimeAttentionレイヤ

    Attributes:
        params (list): パラメータを保持するリスト
        grads (list): 勾配を保持するリスト
        layers (list): Attentionレイヤのリスト
        attention_weights (list): 各Attentionレイヤの各単語への重みのリスト
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.layers = None
        self.attention_weights = None

    def forward(self, encoder_hiddens, decoder_hiddens):
        """順伝播

        Args:
            encoder_hiddens (ndarray): Encoderの隠れ状態
            decoder_hiddens (ndarray): Decoderの隠れ状態

        Returns:
            ndarray: 各単語の重み
        """
        batch_num, times, hidden_size = decoder_hiddens.shape
        out = np.empty_like(decoder_hiddens)
        self.layers = []
        self.attention_weights = []

        for time in range(times):
            layer = Attention()
            out[:, time, :] = layer.forward(
                encoder_hiddens, decoder_hiddens[:, time, :]
            )
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: Encoderの隠れ状態の勾配,
            ndarray: Decoderの隠れ状態の勾配
        """
        batch_num, times, hidden_size = dout.shape
        dencoder_hiddens = 0
        ddecoder_hiddens = np.empty_like(dout)

        for time in range(times):
            layer = self.layers[time]
            dhiddens, dword_hidden = layer.backward(dout[:, time, :])
            dencoder_hiddens += dhiddens
            ddecoder_hiddens[:, time, :] = dword_hidden

        return dencoder_hiddens, ddecoder_hiddens
