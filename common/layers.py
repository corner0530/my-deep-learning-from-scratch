# coding: utf-8
"""レイヤの実装

Attributes:
    Sigmoid (class): シグモイド関数レイヤ
    Relu (class): ReLUレイヤ
    Affine (class): Affineレイヤ
    Dropout (class): Dropoutレイヤ
    BatchNormalization (class): BatchNormalizationレイヤ
    SoftmaxWithLoss (class): Softmax-with-Lossレイヤ
    Convolution (class): Convolutionレイヤ
    Pooling (class): Poolingレイヤ
    MatMul (class): MatMulレイヤ
"""
import numpy as np

from common.functions import cross_entropy_error, sigmoid, softmax
from common.util import col2im, im2col


class Sigmoid:
    """シグモイド関数レイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        outputs (ndarray): 出力値
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.outputs = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力値

        Returns:
            ndarray: 出力値
        """
        out = sigmoid(inputs)
        self.outputs = out
        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        din = dout * (1.0 - self.outputs) * self.outputs
        return din


class Relu:
    """ReLUレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        outputs (ndarray): 出力値
        mask (ndarray): マスク
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.outputs = None
        self.mask = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力値

        Returns:
            ndarray: 出力値
        """
        self.mask = inputs <= 0
        out = inputs.copy()
        out[self.mask] = 0
        self.outputs = out

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        dout[self.mask] = 0
        din = dout

        return din


class Affine:
    """全結合レイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        inputs (ndarray): 入力値
        input_shape (tuple): 入力データの形状
    """

    def __init__(self, weight, bias):
        self.params = [weight, bias]
        self.grads = [np.zeros_like(weight), np.zeros_like(bias)]
        self.inputs = None
        self.input_shape = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力値

        Returns:
            ndarray: 出力値
        """
        # テンソル対応
        self.input_shape = inputs.shape
        self.inputs = inputs.reshape(inputs.shape[0], -1)

        weight, bias = self.params
        out = np.dot(self.inputs, weight) + bias
        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        weight = self.params[0]
        din = np.dot(dout, weight.T)
        dweight = np.dot(self.inputs.T, dout)
        dbias = np.sum(dout, axis=0)

        self.grads[0][...] = dweight
        self.grads[1][...] = dbias

        din = din.reshape(*self.input_shape)  # 入力データの形状に戻す（テンソル対応）
        return din


class Dropout:
    """Dropoutレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        mask (ndarray): マスク
        dropout_ratio (float): Dropoutの割合
    """

    def __init__(self, dropout_ratio=0.5):
        """コンストラクタ

        Args:
            dropout_ratio (float, optional): Dropoutの割合
        """
        self.params = []
        self.grads = []
        self.mask = None
        self.dropout_ratio = dropout_ratio

    def forward(self, inputs, is_train=True):
        """順伝播

        Args:
            inputs (ndarray): 入力値
            is_train (bool, optional): 学習時はTrue. Defaults to True.

        Returns:
            ndarray: 出力値
        """
        if is_train:
            self.mask = np.random.rand(*inputs.shape) > self.dropout_ratio
            out = inputs * self.mask
        else:
            out = inputs * (1.0 - self.dropout_ratio)

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        din = dout * self.mask
        return din


class BatchNormalization:
    """BatchNormalizationレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        gamma (ndarray): スケール係数
        beta (ndarray): シフト係数
        momentum (float): モーメンタム
        running_mean (ndarray): テスト時に使用する平均
        running_var (ndarray): テスト時に使用する分散
        input_shape (tuple): 入力データの形状
        batch_size (int): ミニバッチサイズ
        xc (ndarray): 入力値から平均を引いた値
        std (ndarray): 標準偏差
        dgamma (ndarray): gammaの勾配
        dbeta (ndarray): betaの勾配
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        """コンストラクタ

        Args:
            gamma (ndarray): スケール係数
            beta (ndarray): シフト係数
            momentum (float, optional): モーメンタム
            running_mean (ndarray, optional): テスト時に使用する平均
            running_var (ndarray, optional): テスト時に使用する分散
        """
        self.params = []
        self.grads = []
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, inputs, is_train=True):
        """順伝播

        Args:
            inputs (ndarray): 入力値
            is_train (bool, optional): 学習時はTrue

        Returns:
            ndarray: 出力値
        """
        # テンソル対応
        self.input_shape = inputs.shape
        if inputs.ndim != 2:
            input_num = inputs.shape[0]
            inputs = inputs.reshape(input_num, -1)

        out = self.__forward(inputs, is_train)

        return out.reshape(*self.input_shape)

    def __forward(self, inputs, is_train):
        """順伝播

        Args:
            inputs (ndarray): 入力値(2次元)
            is_train (bool): 学習時はTrue

        Returns:
            ndarray: 出力値(2次元)
        """
        if self.running_mean is None:
            dim = inputs.shape[1]
            self.running_mean = np.zeros(dim)
            self.running_var = np.zeros(dim)

        if is_train:
            # 学習時は正規化
            mu = np.mean(inputs, axis=0)
            xc = inputs - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 1e-7)
            xn = xc / std

            # backward時に使用するデータ
            self.batch_size = inputs.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.var = var

            # テスト時に使用する平均と分散
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            # テスト時
            xc = inputs - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 1e-7)))

        out = self.gamma * xn + self.beta

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        # テンソル対応
        if dout.ndim != 2:
            dout_num = dout.shape[0]
            dout = dout.reshape(dout_num, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配(2次元)

        Returns:
            ndarray: 下流に伝える勾配(2次元)
        """
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / self.var, axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += 2.0 * self.xc * dvar / self.batch_size
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class SoftmaxWithLoss:
    """ソフトマックス関数と交差エントロピー誤差を組み合わせたレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        outputs (ndarray): softmaxの出力
        labels (ndarray): 教師ラベル
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.outputs = None
        self.labels = None

    def forward(self, inputs, labels):
        """順伝播

        Args:
            inputs (ndarray): 入力値
            labels (ndarray): 教師ラベル

        Returns:
            ndarray: 出力値
        """
        self.labels = labels
        self.outputs = softmax(inputs)

        # 教師ラベルがone-hot-vectorの場合、正解のインデックスに変換
        if self.labels.size == self.outputs.size:
            self.labels = self.labels.argmax(axis=1)

        # 損失の計算
        loss = cross_entropy_error(self.outputs, self.labels)
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (int): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        batch_size = self.labels.shape[0]

        din = self.outputs.copy()
        din[np.arange(batch_size), self.labels] -= 1
        din *= dout
        din /= batch_size

        return din


class Convolution:
    """畳み込みレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        stride (int): ストライド
        pad (int): パディング
        inputs (ndarray): 入力値
        col (ndarray): 入力値をim2colで変換した値
        col_w (ndarray): 重みをim2colで変換した値
        dweight (ndarray): 重みの勾配
        dbias (ndarray): バイアスの勾配
    """

    def __init__(self, weight, bias, stride=1, pad=0):
        """コンストラクタ

        Args:
            weight (ndarray): 重み
            bias (ndarray): バイアス
            stride (int): ストライド
            pad (int): パディング
        """
        self.params = [weight, bias]
        self.grads = [np.zeros_like(weight), np.zeros_like(bias)]
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.inputs = None
        self.col = None
        self.col_weight = None

        # 重み・バイアスの勾配
        self.dweight = None
        self.dbias = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力値

        Returns:
            ndarray: 出力値
        """
        weight, bias = self.params
        filter_num, _, filter_height, filter_width = weight.shape
        img_num, _, img_height, img_width = inputs.shape
        out_height = 1 + int((img_height + 2 * self.pad - filter_height) / self.stride)
        out_width = 1 + int((img_width + 2 * self.pad - filter_width) / self.stride)

        # im2col
        col = im2col(inputs, filter_height, filter_width, self.stride, self.pad)
        col_weight = weight.reshape(filter_num, -1).T

        # 行列積
        out = np.dot(col, col_weight) + bias

        # 行列を4次元に戻す
        out = out.reshape(img_num, out_height, out_width, -1).transpose(0, 3, 1, 2)

        self.inputs = inputs
        self.col = col
        self.col_weight = col_weight

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        weight, bias = self.params
        filter_num, filter_channel, filter_height, filter_width = weight.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, filter_num)

        # バイアスの勾配
        self.dbias = np.sum(dout, axis=0)

        # 重みの勾配
        dweight = np.dot(self.col.T, dout)
        self.dweight = dweight.transpose(1, 0).reshape(
            filter_num, filter_channel, filter_height, filter_width
        )

        # 入力値の勾配
        dcol = np.dot(dout, self.col_weight.T)
        din = col2im(
            dcol, self.inputs.shape, filter_height, filter_width, self.stride, self.pad
        )

        return din


class Pooling:
    """プーリングレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        stride (int): ストライド
        pad (int): パディング
        pool_height (int): プーリング領域の高さ
        pool_width (int): プーリング領域の幅
        inputs (ndarray): 入力値
        col (ndarray): 入力値をim2colで変換した値
        arg_max (ndarray): 最大値のインデックス
    """

    def __init__(self, pool_height, pool_width, stride=1, pad=0):
        """コンストラクタ

        Args:
            pool_height (int): プーリング領域の高さ
            pool_width (int): プーリング領域の幅
            stride (int): ストライド
            pad (int): パディング
        """
        self.params = []
        self.grads = []
        self.stride = stride
        self.pad = pad

        # プーリング領域の高さ・幅
        self.pool_height = pool_height
        self.pool_width = pool_width

        # 中間データ（backward時に使用）
        self.inputs = None
        self.col = None
        self.arg_max = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力値

        Returns:
            ndarray: 出力値
        """
        img_num, img_channel, img_height, img_width = inputs.shape
        out_height = 1 + int((img_height - self.pool_height) / self.stride)
        out_width = 1 + int((img_width - self.pool_width) / self.stride)

        # im2col
        col = im2col(inputs, self.pool_height, self.pool_width, self.stride, self.pad)
        col = col.reshape(-1, self.pool_height * self.pool_width)

        # 最大値のインデックスを取得
        arg_max = np.argmax(col, axis=1)

        # 最大値を取得
        out = np.max(col, axis=1)
        out = out.reshape(img_num, out_height, out_width, img_channel).transpose(
            0, 3, 1, 2
        )

        self.inputs = inputs
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_height * self.pool_width
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        din = col2im(
            dcol,
            self.inputs.shape,
            self.pool_height,
            self.pool_width,
            self.stride,
            self.pad,
        )

        return din


class MatMul:
    """行列積レイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        inputs (ndarray): 入力値
    """

    def __init__(self, weight):
        """コンストラクタ

        Args:
            weight (ndarray): 重み
        """
        self.params = [weight]
        self.grads = [np.zeros_like(weight)]
        self.inputs = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力値

        Returns:
            ndarray: 出力値
        """
        self.inputs = inputs
        weight = self.params[0]
        return np.dot(inputs, weight)

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        weight = self.params[0]
        din = np.dot(dout, weight.T)
        dweight = np.dot(self.inputs.T, dout)
        self.grads[0][...] = dweight  # 配列のメモリ位置を固定したうえで上書き(deep copy)
        return din


class Embedding:
    """単語の分散表現を格納するレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        idx (ndarray): 抽出する行のインデックス(単語ID)の配列
    """

    def __init__(self, weight):
        """コンストラクタ

        Args:
            weight (ndarray): 重み
        """
        self.params = [weight]
        self.grads = [np.zeros_like(weight)]
        self.idx = None  # 抽出する行のインデックス(単語ID)の配列

    def forward(self, idx):
        """順伝播

        Args:
            idx (ndarray): 抽出する行のインデックス(単語ID)の配列

        Returns:
            ndarray: 抽出した単語の分散表現
        """
        (weight,) = self.params
        self.idx = idx
        out = weight[idx]  # 特定の行を抜き出す
        return out

    def backward(self, dout):
        """逆伝播

        Args:
            dout (ndarray): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        dweight = self.grads
        dweight[...] = 0  # 形状を保ったまま0に

        np.add.at(dweight, self.idx, dout)  # インデックスが重複しているときは加算するため
        # または
        # for i, word_id in enumerate(self.idx):
        #     dweight[word_id] += dout[i]

        return None


class SigmoidWithLoss:
    """シグモイド関数と交差エントロピー誤差を合わせたレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        loss (float): 損失関数の値
        sigmoid_out (ndarray): シグモイド関数の出力
        labels (ndarray): 教師データ
    """

    def __init__(self):
        """コンストラクタ"""
        self.params = []
        self.grads = []
        self.loss = None
        self.sigmoid_out = None  # sigmoidの出力
        self.labels = None  # 教師データ

    def forward(self, inputs, labels):
        """順伝播

        Args:
            inputs (ndarray): 入力値
            labels (ndarray): 教師データ

        Returns:
            float: 損失関数の値
        """
        self.labels = labels
        self.sigmoid_out = sigmoid(inputs)

        self.loss = cross_entropy_error(
            np.c_[1 - self.sigmoid_out, self.sigmoid_out], self.labels
        )  # シグモイドの出力と1からそれを引いたもの横方向にを結合
        return self.loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (int, optional): 上流から伝わってきた勾配

        Returns:
            ndarray: 下流に伝える勾配
        """
        batch_size = self.labels.shape[0]

        din = (self.sigmoid_out - self.labels) * dout / batch_size
        return din
