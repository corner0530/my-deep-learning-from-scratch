import numpy as np

from common.base_model import BaseModel
from common.time_layers import TimeAffine, TimeEmbedding, TimeLSTM, TimeSoftmaxWithLoss


class Encoder:
    """エンコーダ

    Attributes:
        embed (TimeEmbedding): 単語埋め込みレイヤ
        lstm (TimeLSTM): LSTMレイヤ
        params (list): 重みパラメータのリスト
        grads (list): 勾配のリスト
        hiddens (ndarray): 隠れ状態の配列
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数(文字の種類数)
            wordvec_size (int): 単語ベクトルの次元数
            hidden_size (int): 隠れ状態ベクトルの次元数
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

        # レイヤの生成
        self.embed = TimeEmbedding(embed_weight)
        self.lstm = TimeLSTM(lstm_weight_in, lstm_weight_hid, lstm_bias, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hiddens = None

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力

        Returns:
            ndarray: 最後の時刻の隠れ状態
        """
        inputs = self.embed.forward(inputs)
        hiddens = self.lstm.forward(inputs)
        self.hiddens = hiddens
        return hiddens[:, -1, :]

    def backward(self, dhidden):
        """逆伝播

        Args:
            dhidden (ndarray): LSTMレイヤの最後の隠れ状態に対する勾配

        Returns:
            ndarray: 入力の勾配
        """
        dhiddens = np.zeros_like(self.hiddens)
        dhiddens[:, -1, :] = dhidden

        dout = self.lstm.backward(dhiddens)
        dout = self.embed.backward(dout)

        return dout


class Decoder:
    """デコーダ

    Attributes:
        embed (TimeEmbedding): 単語埋め込みレイヤ
        lstm (TimeLSTM): LSTMレイヤ
        affine (TimeAffine): Affineレイヤ
        params (list): 重みパラメータのリスト
        grads (list): 勾配のリスト
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数(文字の種類数)
            wordvec_size (int): 単語ベクトルの次元数
            hidden_size (int): 隠れ状態ベクトルの次元数
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
        self.embed = TimeEmbedding(embed_weight)
        self.lstm = TimeLSTM(lstm_weight_in, lstm_weight_hid, lstm_bias, stateful=True)
        self.affine = TimeAffine(affine_weight, affine_bias)

        self.params = []
        self.grads = []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, inputs, hidden):
        """順伝播

        Args:
            inputs (ndarray): 入力
            hidden (ndarray): 隠れ状態

        Returns:
            ndarray: 出力
        """
        self.lstm.set_state(hidden)

        out = self.embed.forward(inputs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        """逆伝播

        Args:
            dscore (ndarray): Softmax with Lossレイヤからの勾配

        Returns:
            ndarray: Time LSTMレイヤの時間方向の勾配
        """
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dhidden = self.lstm.dhidden
        return dhidden

    def generate(self, hidden, start_id, sample_size):
        """文章生成

        Args:
            hidden (ndarray): 隠れ状態
            start_id (int): 開始文字のID
            sample_size (int): 生成する文字数

        Returns:
            list: 生成された文字IDのリスト
        """
        sampled = []
        sample_id = start_id
        self.lstm.set_state(hidden)  # Encoderの出力をDecoderの初期状態とする

        for _ in range(sample_size):
            # 文字を1つずつ与え，Affineレイヤが出力するスコアから最大値を持つ文字IDを選ぶ
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2seq(BaseModel):
    """Seq2seqモデル

    Attributes:
        encoder (Encoder): エンコーダ
        decoder (Decoder): デコーダ
        softmax (TimeSoftmaxWithLoss): Softmax with Lossレイヤ
        params (list): 重みパラメータのリスト
        grads (list): 勾配のリスト
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数(文字の種類数)
            wordvec_size (int): 単語ベクトルの次元数
            hidden_size (int): 隠れ状態ベクトルの次元数
        """
        self.encoder = Encoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = Decoder(vocab_size, wordvec_size, hidden_size)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, inputs, labels):
        """順伝播

        Args:
            inputs (ndarray): 入力
            labels (ndarray): 正解ラベル

        Returns:
            float: 損失
        """
        decoder_inputs = labels[:, :-1]
        decoder_labels = labels[:, 1:]

        hidden = self.encoder.forward(inputs)
        score = self.decoder.forward(decoder_inputs, hidden)
        loss = self.softmax.forward(score, decoder_labels)
        return loss

    def backward(self, dout=1):
        """逆伝播

        Args:
            dout (float, optional): 損失の勾配

        Returns:
            ndarray: 入力の勾配
        """
        dout = self.softmax.backward(dout)
        dhidden = self.decoder.backward(dout)
        dout = self.encoder.backward(dhidden)
        return dout

    def generate(self, inputs, start_id, sample_size):
        """文章生成

        Args:
            inputs (ndarray): 入力
            start_id (int): 開始文字のID
            sample_size (int): 生成する文字数

        Returns:
            list: 生成された文字IDのリスト
        """
        hidden = self.encoder.forward(inputs)
        sampled = self.decoder.generate(hidden, start_id, sample_size)
        return sampled
