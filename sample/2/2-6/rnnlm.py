"""RNN言語モデル"""
from common.base_model import BaseModel
from common.np import np
from common.time_layers import TimeAffine, TimeEmbedding, TimeLSTM, TimeSoftmaxWithLoss


class Rnnlm(BaseModel):
    """RNN言語モデル

    Attributes:
        layers (list): レイヤ
        loss_layer (TimeSoftmaxWithLoss): 損失関数レイヤ
        lstm_layer (TimeLSTM): LSTMレイヤ
        params (list): パラメータ
        grads (list): 勾配
    """
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        """コンストラクタ

        Args:
            vocab_size (int, optional): 語彙数
            wordvec_size (int, optional): 単語の分散表現の次元数
            hidden_size (int, optional): 隠れ状態ベクトルの次元数
        """
        # 重みの初期化
        embed_weight = (np.random.randn(vocab_size, wordvec_size) / 100).astype("f")
        lstm_weight_in = (
            np.random.randn(wordvec_size, 4 * hidden_size) / np.sqrt(wordvec_size)
        ).astype("f")
        lstm_weight_hid = (
            np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
        ).astype("f")
        lstm_bias = np.zeros(4 * hidden_size).astype("f")
        affine_weight = (
            np.random.randn(hidden_size, vocab_size) / np.sqrt(hidden_size)
        ).astype("f")
        affine_bias = np.zeros(vocab_size).astype("f")

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_weight),
            TimeLSTM(lstm_weight_in, lstm_weight_hid, lstm_bias, stateful=True),
            TimeAffine(affine_weight, affine_bias),
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # 全ての重みと勾配をリストにまとめる
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, inputs):
        """Softmaxレイヤの直前までの処理

        Args:
            inputs (ndarray): 入力

        Returns:
            ndarray: 出力
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def forward(self, inputs, labels):
        """順伝播

        Args:
            inputs (ndarray): 入力
            labels (ndarray): 教師ラベル

        Returns:
            ndarray: 損失関数の値
        """
        score = self.predict(inputs)
        loss = self.loss_layer.forward(score, labels)
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (int, optional): 上流から伝わってきた勾配

        Returns:
            ndarray: 入力に対する勾配
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        """隠れ状態をリセットする"""
        self.lstm_layer.reset_state()
