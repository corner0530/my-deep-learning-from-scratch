"""時系列レイヤ

Attributes:
    RNN (class): RNNレイヤ
    TimeRNN (class): 時系列版RNNレイヤ
    TimeEmbedding (class): 時系列版Embeddingレイヤ
    TimeAffine (class): 時系列版Affineレイヤ
    TimeSoftmaxWithLoss (class): 時系列版SoftmaxWithLossレイヤ
    LSTM (class): LSTMレイヤ
    TimeLSTM (class): 時系列版LSTMレイヤ
    TimeDropout (class): 時系列版Dropoutレイヤ
    GRU (class): GRUレイヤ
    TimeGRU (class): 時系列版GRUレイヤ
    TimeBiLSTM (class): 時系列版双方向LSTMレイヤ
    TimeSigmoidWithLoss (class): 時系列版SigmoidWithLossレイヤ
"""
from common.functions import sigmoid, softmax
from common.layers import Embedding, SigmoidWithLoss
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


class LSTM:
    """LSTMレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        cache (list): 中間データ
    """

    def __init__(self, weight_in, weight_hid, bias):
        """コンストラクタ

        Args:
            weight_x (ndarray): 入力に対する重み
            weight_h (ndarray): 隠れ状態に対する重み
            bias (ndarray): バイアス
        """
        self.params = [weight_in, weight_hid, bias]
        self.grads = [
            np.zeros_like(weight_in),
            np.zeros_like(weight_hid),
            np.zeros_like(bias),
        ]
        self.cache = None

    def forward(self, inputs, hidden_prev, cell_prev):
        """順伝播

        Args:
            inputs (ndarray): 入力
            hidden_prev (ndarray): 前時刻の隠れ状態
            cell_prev (ndarray): 前時刻のセル状態

        Returns:
            ndarray: 隠れ状態,
            ndarray: セル状態
        """
        weight_in, weight_hid, bias = self.params
        batch_num, hidden_dim = hidden_prev.shape

        # 4つのパラメータを一括でアフィン返還
        affine = np.dot(inputs, weight_in) + np.dot(hidden_prev, weight_hid) + bias

        # slice
        forget = affine[:, :hidden_dim]
        new_cell = affine[:, hidden_dim : 2 * hidden_dim]
        input = affine[:, 2 * hidden_dim : 3 * hidden_dim]
        output = affine[:, 3 * hidden_dim :]

        forget = sigmoid(forget)
        new_cell = np.tanh(new_cell)
        input = sigmoid(input)
        output = sigmoid(output)

        cell_next = forget * cell_prev + new_cell * input
        hidden_next = output * np.tanh(cell_next)

        self.cache = (
            inputs,
            hidden_prev,
            cell_prev,
            input,
            forget,
            new_cell,
            output,
            cell_next,
        )
        return hidden_next, cell_next

    def backward(self, dhidden_next, dcell_next):
        """逆伝播

        Args:
            dhidden_next (ndarray): 次時刻の隠れ状態の勾配
            dcell_next (ndarray): 次時刻のセル状態の勾配

        Returns:
            ndarray: 入力の勾配
            ndarray: 隠れ状態の勾配
            ndarray: セル状態の勾配
        """
        weight_in, weight_hid, bias = self.params
        (
            inputs,
            hidden_prev,
            cell_prev,
            input,
            forget,
            new_cell,
            output,
            cell_next,
        ) = self.cache

        tanh_cell_next = np.tanh(cell_next)

        dsum = dcell_next + (dhidden_next * output) * (
            1 - tanh_cell_next * tanh_cell_next
        )

        dcell_prev = dsum * forget

        dinput = dsum * new_cell
        dforget = dsum * cell_prev
        doutput = dhidden_next * tanh_cell_next
        dnew_cell = dsum * input

        dinput *= input * (1 - input)
        dforget *= forget * (1 - forget)
        doutput *= output * (1 - output)
        dnew_cell *= 1 - new_cell * new_cell

        # 横方向に連結
        daffine = np.hstack((dforget, dnew_cell, dinput, doutput))

        dweight_hid = np.dot(hidden_prev.T, daffine)
        dweight_in = np.dot(inputs.T, daffine)
        dbias = np.sum(daffine, axis=0)

        self.grads[0][...] = dweight_in
        self.grads[1][...] = dweight_hid
        self.grads[2][...] = dbias

        din = np.dot(daffine, weight_in.T)
        dhidden_prev = np.dot(daffine, weight_hid.T)

        return din, dhidden_prev, dcell_prev


class TimeLSTM:
    """TimeLSTMレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        layers (list): LSTMレイヤのリスト
        hidden (ndarray): 隠れ状態
        cell (ndarray): セル状態
        dhidden (ndarray): 隠れ状態の勾配
        stateful (bool): 隠れ状態を維持するかどうか
    """

    def __init__(self, weight_in, weight_hid, bias, stateful=False):
        """コンストラクタ

        Args:
            weight_in (ndarray): 入力に対する重み
            weight_hid (ndarray): 隠れ状態に対する重み
            bias (ndarray): バイアス
            stateful (bool, optional): 隠れ状態を維持するかどうか
        """
        self.params = [weight_in, weight_hid, bias]
        self.grads = [
            np.zeros_like(weight_in),
            np.zeros_like(weight_hid),
            np.zeros_like(bias),
        ]
        self.layers = None

        self.hidden = None
        self.cell = None
        self.dhidden = None
        self.stateful = stateful

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力

        Returns:
            ndarray: 隠れ状態
        """
        weight_in, weight_hid, bias = self.params
        batch_num, times, input_dim = inputs.shape
        hidden_dim = weight_hid.shape[0]

        self.layers = []
        hiddens = np.empty((batch_num, times, hidden_dim), dtype="f")

        if not self.stateful or self.hidden is None:
            self.hidden = np.zeros((batch_num, hidden_dim), dtype="f")
        if not self.stateful or self.cell is None:
            self.cell = np.zeros((batch_num, hidden_dim), dtype="f")

        for time in range(times):
            layer = LSTM(*self.params)
            self.hidden, self.cell = layer.forward(
                inputs[:, time, :], self.hidden, self.cell
            )
            hiddens[:, time, :] = self.hidden

            self.layers.append(layer)

        return hiddens

    def backward(self, dhiddens):
        """逆伝播

        Args:
            dhiddens (ndarray): 次時刻の隠れ状態の勾配

        Returns:
            ndarray: 入力の勾配
        """
        weight_in, weight_hid, bias = self.params
        batch_num, times, hidden_dim = dhiddens.shape
        input_dim = weight_in.shape[0]

        dinputs = np.empty((batch_num, times, input_dim), dtype="f")
        dhidden = 0
        dcell = 0

        grads = [0, 0, 0]
        for time in reversed(range(times)):
            layer = self.layers[time]
            dinput, dhidden, dcell = layer.backward(
                dhiddens[:, time, :] + dhidden, dcell
            )
            dinputs[:, time, :] = dinput
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dhidden = dhidden
        return dinputs

    def set_state(self, hidden, cell=None):
        """隠れ状態・セル状態を設定

        Args:
            hidden (ndarray): 隠れ状態
            cell (ndarray, optional): セル状態

        """
        self.hidden = hidden
        self.cell = cell

    def reset_state(self):
        """隠れ状態・セル状態をリセット"""
        self.hidden = None
        self.cell = None


class TimeDropout:
    """TimeDropoutレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        dropout_ratio (float): ドロップアウトの割合
        mask (ndarray): マスク
        train_flg (bool): 学習フラグ
    """

    def __init__(self, dropout_ratio=0.5):
        """コンストラクタ

        Args:
            dropout_ratio (float, optional): ドロップアウトの割合
        """
        self.params = []
        self.grads = []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力

        Returns:
            ndarray: 出力
        """
        if self.train_flg:
            flg = np.random.rand(*inputs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return inputs * self.mask
        else:
            return inputs

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 次レイヤからの勾配

        Returns:
            ndarray: 前レイヤへの勾配
        """
        return dout * self.mask


class GRU:
    """GRUレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        cache (list): 中間データ
    """

    def __init__(self, weight_in, weight_hidden, bias):
        """コンストラクタ

        Args:
            weight_in (ndarray): 入力に対する重み
            weight_hidden (ndarray): 隠れ状態に対する重み
            bias (ndarray): バイアス
        """
        self.params = [weight_in, weight_hidden, bias]
        self.grads = [
            np.zeros_like(weight_in),
            np.zeros_like(weight_hidden),
            np.zeros_like(bias),
        ]
        self.cache = None

    def forward(self, input, hidden_prev):
        """順伝播

        Args:
            input (ndarray): 入力
            hidden_prev (ndarray): 前時刻の隠れ状態

        Returns:
            ndarray: 隠れ状態
        """
        weight_in, weight_hidden, bias = self.params
        hidden_size = weight_hidden.shape[0]
        weight_in_update = weight_in[:, :hidden_size]
        weight_in_reset = weight_in[:, hidden_size : 2 * hidden_size]
        weight_in_hidden = weight_in[:, 2 * hidden_size :]
        weight_hidden_update = weight_hidden[:, :hidden_size]
        weight_hidden_reset = weight_hidden[:, hidden_size : 2 * hidden_size]
        weight_hidden_hidden = weight_hidden[:, 2 * hidden_size :]
        bias_update = bias[:hidden_size]
        bias_reset = bias[hidden_size : 2 * hidden_size]
        bias_hidden = bias[2 * hidden_size :]

        update = sigmoid(
            np.dot(input, weight_in_update)
            + np.dot(hidden_prev, weight_hidden_update)
            + bias_update
        )
        reset = sigmoid(
            np.dot(input, weight_in_reset)
            + np.dot(hidden_prev, weight_hidden_reset)
            + bias_reset
        )
        hidden_hat = np.tanh(
            np.dot(input, weight_in_hidden)
            + np.dot(reset * hidden_prev, weight_hidden_hidden)
            + bias_hidden
        )
        hidden_next = (1 - update) * hidden_prev + update * hidden_hat

        self.cache = (input, hidden_prev, update, reset, hidden_hat)

        return hidden_next

    def backward(self, dhidden_next):
        """逆伝播

        Args:
            dhidden_next (ndarray): 次レイヤからの勾配

        Returns:
            ndarray: 前レイヤへの勾配
        """
        weight_in, weight_hidden, bias = self.params
        hidden_size = weight_hidden.shape[0]
        weight_in_update = weight_in[:, :hidden_size]
        weight_in_reset = weight_in[:, hidden_size : 2 * hidden_size]
        weight_in_hidden = weight_in[:, 2 * hidden_size :]
        weight_hidden_update = weight_hidden[:, :hidden_size]
        weight_hidden_reset = weight_hidden[:, hidden_size : 2 * hidden_size]
        weight_hidden_hidden = weight_hidden[:, 2 * hidden_size :]
        input, hidden_prev, update, reset, hidden_hat = self.cache

        dhidden_hat = dhidden_next * update
        dhidden_prev = dhidden_next * (1 - update)

        # tanh
        dt = dhidden_hat * (1 - hidden_hat * hidden_hat)
        dbias_hidden = np.sum(dt, axis=0)
        dweight_hidden_hidden = np.dot((reset * hidden_prev).T, dt)
        dhidden_reset = np.dot(dt, weight_hidden_hidden.T)
        dweight_in_hidden = np.dot(input.T, dt)
        din = np.dot(dt, weight_in_hidden.T)
        dhidden_prev += reset * dhidden_reset

        # update gate
        dupdate = dhidden_next * hidden_hat - dhidden_next * hidden_prev
        dt = dupdate * update * (1 - update)
        dbias_update = np.sum(dt, axis=0)
        dweight_hidden_update = np.dot(hidden_prev.T, dt)
        dhidden_prev += np.dot(dt, weight_hidden_update.T)
        dweight_in_update = np.dot(input.T, dt)
        din += np.dot(dt, weight_in_update.T)

        # reset gate
        dreset = dhidden_reset * hidden_prev
        dt = dreset * reset * (1 - reset)
        dbias_reset = np.sum(dt, axis=0)
        dweight_hidden_reset = np.dot(hidden_prev.T, dt)
        dhidden_prev += np.dot(dt, weight_hidden_reset.T)
        dweight_in_reset = np.dot(input.T, dt)
        din += np.dot(dt, weight_in_reset.T)

        self.dweight_in = np.hstack(
            (dweight_in_update, dweight_in_reset, dweight_in_hidden)
        )
        self.dweight_hidden = np.hstack(
            (dweight_hidden_update, dweight_hidden_reset, dweight_hidden_hidden)
        )
        self.dbias = np.hstack((dbias_update, dbias_reset, dbias_hidden))

        self.grads[0][...] = self.dweight_in
        self.grads[1][...] = self.dweight_hidden
        self.grads[2][...] = self.dbias

        return din, dhidden_prev


class TimeGRU:
    """TimeGRUレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        layers (list): GRUレイヤのリスト
        hidden (ndarray): 隠れ状態
        dhidden (ndarray): 勾配
        stateful (bool): 隠れ状態を維持するかどうか
    """

    def __init__(self, weight_in, weight_hidden, bias, stateful=False):
        """コンストラクタ

        Args:
            weight_in (ndarray): 入力に対する重み
            weight_hidden (ndarray): 隠れ状態に対する重み
            bias (ndarray): バイアス
            stateful (bool, optional): 隠れ状態を維持するかどうか
        """
        self.params = [weight_in, weight_hidden, bias]
        self.grads = [
            np.zeros_like(weight_in),
            np.zeros_like(weight_hidden),
            np.zeros_like(bias),
        ]
        self.layers = None
        self.hidden = None
        self.dhidden = None
        self.stateful = stateful

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力

        Returns:
            ndarray: 最後の時刻の隠れ状態
        """
        weight_in, weight_hidden, bias = self.params
        batch_num, times, input_dim = inputs.shape
        hidden_size = weight_hidden.shape[0]

        self.layers = []
        hiddens = np.empty((batch_num, times, hidden_size), dtype="f")

        if not self.stateful or self.hidden is None:
            self.hidden = np.zeros((batch_num, hidden_size), dtype="f")

        for time in range(times):
            layer = GRU(*self.params)
            self.hidden = layer.forward(inputs[:, time, :], self.hidden)
            hiddens[:, time, :] = self.hidden
            self.layers.append(layer)

        return hiddens

    def backward(self, dhiddens):
        """逆伝播

        Args:
            dhiddens (ndarray): 隠れ状態に対する勾配

        Returns:
            ndarray: 入力に対する勾配
        """
        weight_in, weight_hidden, bias = self.params
        batch_num, times, hidden_size = dhiddens.shape
        input_dim = weight_in.shape[0]

        dinputs = np.empty((batch_num, times, input_dim), dtype="f")
        dhidden = 0
        grads = [0, 0, 0]

        for time in reversed(range(times)):
            layer = self.layers[time]
            dinput, dhidden = layer.backward(dhiddens[:, time, :] + dhidden)
            dinputs[:, time, :] = dinput

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dhidden = dhidden
        return dinputs

    def set_state(self, hidden):
        """隠れ状態を設定

        Args:
            hidden (ndarray): 隠れ状態
        """
        self.hidden = hidden

    def reset_state(self):
        """隠れ状態をリセット"""
        self.hidden = None


class TimeBiLSTM:
    """TimeBiLSTMレイヤ

    Attributes:
        forward_lstm (TimeLSTM): 順伝播用LSTMレイヤ
        backward_lstm (TimeLSTM): 逆伝播用LSTMレイヤ
        params (list): パラメータ
        grads (list): 勾配
    """

    def __init__(
        self,
        weight_in1,
        weight_hidden1,
        bias1,
        weight_in2,
        weight_hidden2,
        bias2,
        stateful=False,
    ):
        """コンストラクタ

        Args:
            weight_in1 (ndarray): 順伝播用LSTMレイヤの入力に対する重み
            weight_hidden1 (ndarray): 順伝播用LSTMレイヤの隠れ状態に対する重み
            bias1 (ndarray): 順伝播用LSTMレイヤのバイアス
            weight_in2 (ndarray): 逆伝播用LSTMレイヤの入力に対する重み
            weight_hidden2 (ndarray): 逆伝播用LSTMレイヤの隠れ状態に対する重み
            bias2 (ndarray): 逆伝播用LSTMレイヤのバイアス
            stateful (bool, optional): 隠れ状態を維持するかどうか
        """
        self.forward_lstm = TimeLSTM(weight_in1, weight_hidden1, bias1, stateful)
        self.backward_lstm = TimeLSTM(weight_in2, weight_hidden2, bias2, stateful)
        self.params = self.forward_lstm.params + self.backward_lstm.params
        self.grads = self.forward_lstm.grads + self.backward_lstm.grads

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力

        Returns:
            ndarray: 隠れ状態
        """
        out1 = self.forward_lstm.forward(inputs)
        out2 = self.backward_lstm.forward(inputs[:, ::-1])
        out2 = out2[:, ::-1]

        out = np.concatenate((out1, out2), axis=2)
        return out

    def backward(self, dhiddens):
        """逆伝播

        Args:
            dhiddens (ndarray): 隠れ状態に対する勾配

        Returns:
            ndarray: 入力に対する勾配
        """
        hidden_size = dhiddens.shape[2] // 2
        dout1 = dhiddens[:, :, :hidden_size]
        dout2 = dhiddens[:, :, hidden_size:]

        dinputs1 = self.forward_lstm.backward(dout1)
        dout2 = dout2[:, ::-1]
        dinputs2 = self.backward_lstm.backward(dout2)
        dinputs2 = dinputs2[:, ::-1]
        dinputs = dinputs1 + dinputs2
        return dinputs


class TimeSigmoidWithLoss:
    """TimeSigmoidWithLossレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        inputs_shape (tuple): 入力の形状
        layers (list): SigmoidWithLossレイヤのリスト
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.inputs_shape = None
        self.layers = None

    def forward(self, inputs, labels):
        """順伝播

        Args:
            inputs (ndarray): 入力
            labels (ndarray): 教師データ

        Returns:
            float: 損失
        """
        batch_num, times = inputs.shape
        self.inputs_shape = inputs.shape

        self.layers = []
        loss = 0

        for time in range(times):
            layer = SigmoidWithLoss()
            loss += layer.forward(inputs[:, time], labels[:, time])
            self.layers.append(layer)

        return loss / times

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (float, optional): 出力に対する勾配

        Returns:
            ndarray: 入力に対する勾配
        """
        batch_num, times = self.inputs_shape
        dinputs = np.empty(self.inputs_shape, dtype="f")

        dout *= 1 / times
        for time in range(times):
            layer = self.layers[time]
            dinputs[:, time] = layer.backward(dout)

        return dinputs
