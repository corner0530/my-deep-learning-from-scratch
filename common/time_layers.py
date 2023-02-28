"""時系列レイヤ

Attributes:
    RNN (class): RNNレイヤ
    TimeRNN (class): 時系列版RNNレイヤ
    TimeEmbedding (class): 時系列版Embeddingレイヤ
    TimeAffine (class): 時系列版Affineレイヤ
    TimeSoftmaxWithLoss (class): 時系列版SoftmaxWithLossレイヤ
"""
from common.functions import softmax
from common.layers import Embedding
from common.np import np


class RNN:
    """RNNレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        cache (tuple): 中間データ
    """

    def __init__(self, weight_input, weight_hidden, bias):
        """コンストラクタ

        Args:
            weight_input (ndarray): 入力inputに対する重み
            weight_hidden (ndarray): 中間層hiddenに対する重み
            bias (ndarray): バイアス
        """
        self.params = [weight_input, weight_hidden, bias]
        self.grads = [
            np.zeros_like(weight_input),
            np.zeros_like(weight_hidden),
            np.zeros_like(bias),
        ]
        self.cache = None

    def forward(self, input, hidden_prev):
        """順伝播

        Args:
            input (ndarray): 入力
            hidden_prev (ndarray): 前時刻の中間層

        Returns:
            ndarray: 現時刻の中間層
        """
        weight_input, weight_hidden, bias = self.params
        t = np.dot(hidden_prev, weight_hidden) + np.dot(input, weight_input) + bias
        hidden_next = np.tanh(t)
        self.cache = (input, hidden_prev, hidden_next)
        return hidden_next

    def backward(self, dhidden_next):
        """逆伝播

        Args:
            dhidden_next (ndarray): 現時刻の中間層の勾配

        Returns:
            ndarray: 現時刻の入力の勾配,
            ndarray: 前時刻の中間層の勾配"""
        weight_input, weight_hidden, bias = self.params
        input, hidden_prev, hidden_next = self.cache

        dt = dhidden_next * (1 - hidden_next * hidden_next)
        dbias = np.sum(dt, axis=0)
        dweight_hidden = np.dot(hidden_prev.T, dt)
        dhidden_prev = np.dot(dt, weight_hidden.T)
        dweight_input = np.dot(input.T, dt)
        dinput = np.dot(dt, weight_input.T)

        self.grads[0][...] = dweight_input
        self.grads[1][...] = dweight_hidden
        self.grads[2][...] = dbias

        return dinput, dhidden_prev


class TimeRNN:
    """TimeRNNレイヤ

    RNNレイヤを連結したレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        layers (list): RNNレイヤのリスト
        hidden (ndarray): 最後のRNNレイヤの隠れ状態
        dhidden (ndarray): 最後のRNNレイヤの勾配
        stateful (bool): 隠れ状態を維持するかどうか
    """

    def __init__(self, weight_input, weight_hidden, bias, stateful=False):
        """コンストラクタ

        Args:
            weight_input (ndarray): 入力inputに対する重み
            weight_hidden (ndarray): 中間層hiddenに対する重み
            bias (ndarray): バイアス
            stateful (bool, optional): 隠れ状態を維持するかどうか
        """
        self.params = [weight_input, weight_hidden, bias]
        self.grads = [
            np.zeros_like(weight_input),
            np.zeros_like(weight_hidden),
            np.zeros_like(bias),
        ]
        self.layers = None

        self.hidden = None
        self.dhidden = None
        self.stateful = stateful

    def set_state(self, hidden):
        """隠れ状態を設定

        Args:
            hidden (ndarray): 隠れ状態
        """
        self.hidden = hidden

    def reset_state(self):
        """隠れ状態をリセット"""
        self.hidden = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力(時系列データをまとめたもの)

        Returns:
            ndarray: 出力
        """
        weight_input, weight_hidden, bias = self.params
        batch_size, times, input_dim = inputs.shape
        input_dim, hidden_size = weight_input.shape

        self.layers = []
        hiddens = np.empty((batch_size, times, hidden_size), dtype="f")

        # 初期化
        if not self.stateful or self.hidden is None:
            self.hidden = np.zeros((batch_size, hidden_size), dtype="f")

        # 各時刻の隠れ状態を計算
        for time in range(times):
            layer = RNN(*self.params)
            self.hidden = layer.forward(inputs[:, time, :], self.hidden)
            hiddens[:, time, :] = self.hidden
            self.layers.append(layer)

        return hiddens

    def backward(self, dhiddens):
        """逆伝播

        Args:
            dhiddens (ndarray): 出力の勾配

        Returns:
            ndarray: 入力の勾配
        """
        weight_input, weight_hidden, bias = self.params
        batch_size, times, hidden_size = dhiddens.shape
        input_dim, hidden_size = weight_input.shape

        dinputs = np.empty((batch_size, times, input_dim), dtype="f")
        dhidden = 0
        grads = [0, 0, 0]

        # 逆順に各層の勾配を計算
        for time in reversed(range(times)):
            layer = self.layers[time]
            dinput, dhidden = layer.backward(
                dhiddens[:, time, :] + dhidden
            )  # 各時刻の勾配を計算
            dinputs[:, time, :] = dinput

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dhidden = dhidden

        return dinputs


class TimeEmbedding:
    """TimeEmbeddingレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        layers (list): Embeddingレイヤのリスト
        weight (ndarray): 重み
    """

    def __init__(self, weight):
        """コンストラクタ

        Args:
            weight (ndarray): 重み
        """
        self.params = [weight]
        self.grads = [np.zeros_like(weight)]
        self.layers = None
        self.weight = weight

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力(時系列データをまとめたもの)

        Returns:
            ndarray: 出力
        """
        batch_size, times = inputs.shape
        value_size, hidden_dim = self.weight.shape

        out = np.empty((batch_size, times, hidden_dim), dtype="f")
        self.layers = []

        # 各時刻の出力を計算
        for time in range(times):
            layer = Embedding(self.weight)
            out[:, time, :] = layer.forward(inputs[:, time])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 出力の勾配

        Returns:
            ndarray: 入力の勾配
        """
        batch_size, times, hidden_dim = dout.shape

        # 各時刻の勾配を計算し足し合わせる
        grad = 0
        for time in range(times):
            layer = self.layers[time]
            layer.backward(dout[:, time, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    """TimeAffineレイヤ

    Affineレイヤを連結したレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        input (ndarray): 入力
    """

    def __init__(self, weight, bias):
        """コンストラクタ

        Args:
            weight (ndarray): 重み
            bias (ndarray): バイアス
        """
        self.params = [weight, bias]
        self.grads = [np.zeros_like(weight), np.zeros_like(bias)]
        self.input = None

    def forward(self, input):
        """順伝播

        Args:
            input (ndarray): 入力

        Returns:
            ndarray: 出力
        """
        # 入力
        batch_size, times, input_dim = input.shape
        weight, bias = self.params

        # 全てのAffineレイヤをまとめて行列計算
        rinput = input.reshape(batch_size * times, -1)
        out = np.dot(rinput, weight) + bias
        self.input = input
        return out.reshape(batch_size, times, -1)

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 出力の勾配

        Returns:
            ndarray: 入力の勾配
        """
        # 入力
        input = self.input
        batch_size, times, input_dim = input.shape
        weight, bias = self.params

        # 全てのAffineレイヤをまとめて行列計算
        dout = dout.reshape(batch_size * times, -1)
        rinput = input.reshape(batch_size * times, -1)

        # 勾配
        dbias = np.sum(dout, axis=0)
        dweight = np.dot(rinput.T, dout)
        dinput = np.dot(dout, weight.T).reshape(*input.shape)

        # 勾配を格納
        self.grads[0][...] = dweight
        self.grads[1][...] = dbias

        return dinput


class TimeSoftmaxWithLoss:
    """TimeSoftmaxWithLossレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        cache (ndarray): 中間データ
        ignore_label (int): 無視するラベル
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.cache = None
        self.ignore_label = -1

    def forward(self, inputs, labels):
        """順伝播

        Args:
            inputs (ndarray): 入力
            labels (ndarray): 教師ラベル

        Returns:
            float: 損失関数
        """
        batch_size, times, value_size = inputs.shape

        if labels.ndim == 3:  # 教師ラベルがone-hot vectorの場合
            labels = labels.argmax(axis=2)

        # 無視するラベルのマスク
        mask = labels != self.ignore_label

        # バッチ分と時系列分をまとめる
        inputs = inputs.reshape(batch_size * times, value_size)
        labels = labels.reshape(batch_size * times)
        mask = mask.reshape(batch_size * times)

        # Softmax関数を適用
        outputs = softmax(inputs)
        losses = np.log(outputs[np.arange(batch_size * times), labels])
        losses *= mask
        loss = -np.sum(losses) / np.sum(mask)

        self.cache = (labels, outputs, mask, (batch_size, times, value_size))
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (float): 出力の勾配

        Returns:
            ndarray: 入力の勾配
        """
        labels, outputs, mask, (batch_size, times, value_size) = self.cache

        dinput = outputs
        dinput[np.arange(batch_size * times), labels] -= 1
        dinput = (dinput * dout / np.sum(mask)) * mask[
            :, np.newaxis
        ]  # ignore_labelの勾配は0にする

        dinput = dinput.reshape((batch_size, times, value_size))

        return dinput
