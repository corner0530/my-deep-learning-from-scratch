import numpy as np
from seq2seq import Encoder, Seq2seq

from common.time_layers import TimeAffine, TimeEmbedding, TimeLSTM, TimeSoftmaxWithLoss


class PeekyDecoder:
    """覗き見の改良を加えたデコーダ

    Attributes:
        embed (TimeEmbedding): 単語埋め込みレイヤ
        lstm (TimeLSTM): LSTMレイヤ
        affine (TimeAffine): Affineレイヤ
        params (list): 重みパラメータのリスト
        grads (list): 勾配のリスト
        cache (ndarray): 隠れ状態のサイズ
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
            np.random.randn(hidden_size + wordvec_size, 4 * hidden_size)  # 隠れ状態分を追加
            / np.sqrt(hidden_size + wordvec_size)
        ).astype("f")
        lstm_weight_hid = (
            np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
        ).astype("f")
        lstm_bias = np.zeros(4 * hidden_size).astype("f")
        affine_weight = (
            np.random.randn(hidden_size + hidden_size, vocab_size)  # 隠れ状態分を追加
            / np.sqrt(hidden_size + hidden_size)
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
        self.cache = None

    def forward(self, inputs, hidden):
        """順伝播

        Args:
            inputs (ndarray): 入力
            hidden (ndarray): 隠れ状態

        Returns:
            ndarray: 出力
        """
        batch_num, times = inputs.shape
        batch_num, hidden_size = hidden.shape

        self.lstm.set_state(hidden)

        out = self.embed.forward(inputs)
        hiddens = np.repeat(hidden, times, axis=0).reshape(
            batch_num, times, hidden_size
        )  # 隠れ状態を複製
        out = np.concatenate((hiddens, out), axis=2)  # 隠れ状態と単語埋め込みを結合

        out = self.lstm.forward(out)
        out = np.concatenate((hiddens, out), axis=2)  # 隠れ状態とLSTMの出力を結合

        score = self.affine.forward(out)
        self.cache = hidden_size
        return score

    def backward(self, dscore):
        """逆伝播

        Args:
            dscore (ndarray): Softmax with Lossレイヤからの勾配

        Returns:
            ndarray: Time LSTMレイヤの時間方向の勾配
        """
        hidden_size = self.cache

        dout = self.affine.backward(dscore)
        dout = dout[:, :, hidden_size:]
        dhiddens0 = dout[:, :, :hidden_size]
        dout = self.lstm.backward(dout)
        dembed = dout[:, :, hidden_size:]
        dhiddens1 = dout[:, :, :hidden_size]
        self.embed.backward(dembed)

        dhiddens = dhiddens0 + dhiddens1
        dhidden = self.lstm.dhidden + np.sum(dhiddens, axis=1)
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
        char_id = start_id
        self.lstm.set_state(hidden)

        hidden_size = hidden.shape[1]
        peeky_hidden = hidden.reshape(1, 1, hidden_size)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_hidden, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_hidden, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(int(char_id))

        return sampled


class PeekySeq2seq(Seq2seq):
    """Peeky Seq2seqモデル

    Attributes:
        encoder (Encoder): エンコーダ
        decoder (PeekyDecoder): デコーダ
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
        self.decoder = PeekyDecoder(vocab_size, wordvec_size, hidden_size)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
