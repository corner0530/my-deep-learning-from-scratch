"""改良版RNNLM"""
from common.base_model import BaseModel
from common.np import np
from common.time_layers import (
    TimeAffine,
    TimeDropout,
    TimeEmbedding,
    TimeLSTM,
    TimeSoftmaxWithLoss,
)


class BetterRnnlm(BaseModel):
    """改良版RNNLM

    - TimeLSTMレイヤを2つ使う
    - 重み共有
    - dropout

    Attributes:
        layers (list): レイヤのリスト
        lstm_layers (list): LSTMレイヤのリスト
        drop_layers (list): Dropoutレイヤのリスト
        loss_layer (TimeSoftmaxWithLoss): 損失レイヤ
        params (list): 重みパラメータのリスト
        grads (list): 勾配のリスト
    """
    def __init__(
        self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5
    ):
        """コンストラクタ

        Args:
            vocab_size (int, optional): 語彙数
            wordvec_size (int, optional): 単語ベクトルの次元数
            hidden_size (int, optional): 隠れ状態ベクトルの次元数
            dropout_ratio (float, optional): dropoutの割合
        """
        # 重みの初期化
        embed_weight = (np.random.randn(vocab_size, wordvec_size) / 100).astype("f")
        lstm_weight_in1 = (
            np.random.randn(wordvec_size, 4 * hidden_size) / np.sqrt(wordvec_size)
        ).astype("f")
        lstm_weight_hid1 = (
            np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
        ).astype("f")
        lstm_bias1 = np.zeros(4 * hidden_size).astype("f")
        lstm_weight_in2 = (
            np.random.randn(wordvec_size, 4 * hidden_size) / np.sqrt(wordvec_size)
        ).astype("f")
        lstm_weight_hid2 = (
            np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
        ).astype("f")
        lstm_bias2 = np.zeros(4 * hidden_size).astype("f")
        affine_bias = np.zeros(vocab_size).astype("f")

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_weight),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_weight_in1, lstm_weight_hid1, lstm_bias1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_weight_in2, lstm_weight_hid2, lstm_bias2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_weight.T, affine_bias),  # 重み共有
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, inputs, train_flg=False):
        """予測

        Args:
            inputs (list): 入力
            train_flg (bool, optional): 学習フラグ

        Returns:
            list: 出力
        """
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def forward(self, inputs, labels, train_flg=True):
        """順伝播

        Args:
            inputs (list): 入力
            labels (list): 教師ラベル
            train_flg (bool, optional): 学習フラグ

        Returns:
            list: 損失関数の値
        """
        score = self.predict(inputs, train_flg)
        loss = self.loss_layer.forward(score, labels)
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (list, optional): 上流から伝わってくる勾配

        Returns:
            list: 勾配
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        """隠れ状態のリセット"""
        for layer in self.lstm_layers:
            layer.reset_state()
