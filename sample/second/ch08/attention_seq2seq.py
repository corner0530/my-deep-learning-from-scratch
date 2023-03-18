import numpy as np
from attention_layer import TimeAttention

from common.time_layers import TimeAffine, TimeEmbedding, TimeLSTM, TimeSoftmaxWithLoss
from sample.second.ch07.seq2seq import Encoder, Seq2seq


class AttentionEncoder(Encoder):
    """Attention付きEncoderクラス

    Attributes:
        embed (TimeEmbedding): 単語埋め込みレイヤ
        lstm (TimeLSTM): LSTMレイヤ
        params (list): パラメータを保持するリスト
        grads (list): 勾配を保持するリスト
        hiddens (ndarray): 隠れ状態の配列
    """

    def forward(self, inputs):
        """順伝播

        Args:
            inputs (ndarray): 入力

        Returns:
            ndarray: 隠れ状態の配列
        """
        inputs = self.embed.forward(inputs)
        hiddens = self.lstm.forward(inputs)
        return hiddens  # すべての隠れ状態を返す

    def backward(self, dhiddens):
        """逆伝播

        Args:
            dhiddens (ndarray): 隠れ状態の勾配

        Returns:
            ndarray: 入力に対する勾配
        """
        dout = self.lstm.backward(dhiddens)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:
    """Attention付きDecoderクラス

    Attributes:
        embed (TimeEmbedding): 単語埋め込みレイヤ
        lstm (TimeLSTM): LSTMレイヤ
        attention (TimeAttention): Attentionレイヤ
        affine (TimeAffine): Affineレイヤ
        params (list): パラメータを保持するリスト
        grads (list): 勾配を保持するリスト
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数(文字の種類数)
            wordvec_size (int): 単語ベクトルの次元数
            hidden_size (int): 隠れ状態ベクトルの次元数
        """
        embed_weight = (np.random.randn(vocab_size, wordvec_size) / 100).astype("f")
        lstm_weight_in = (
            np.random.randn(wordvec_size, 4 * hidden_size) / np.sqrt(wordvec_size)
        ).astype("f")
        lstm_weight_hid = (
            np.random.randn(hidden_size, 4 * hidden_size) / np.sqrt(hidden_size)
        ).astype("f")
        lstm_bias = np.zeros(4 * hidden_size).astype("f")
        affine_weight = (
            np.random.randn(2 * hidden_size, vocab_size) / np.sqrt(2 * hidden_size)
        ).astype("f")
        affine_bias = np.zeros(vocab_size).astype("f")

        self.embed = TimeEmbedding(embed_weight)
        self.lstm = TimeLSTM(lstm_weight_in, lstm_weight_hid, lstm_bias, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_weight, affine_bias)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params = []
        self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, inputs, encoder_hiddens):
        """順伝播

        Args:
            inputs (ndarray): 入力
            encoder_hiddens (ndarray): Encoderの隠れ状態の配列

        Returns:
            ndarray: 出力
        """
        word_hidden = encoder_hiddens[:, -1]
        self.lstm.set_state(word_hidden)

        out = self.embed.forward(inputs)
        decoder_hiddens = self.lstm.forward(out)
        context = self.attention.forward(encoder_hiddens, decoder_hiddens)
        out = np.concatenate(
            (context, decoder_hiddens), axis=2
        )  # TimeAttentionレイヤの出力とLSTMレイヤの出力を結合
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        """逆伝播

        Args:
            dscore (ndarray): 出力に対する勾配

        Returns:
            ndarray: 入力に対する勾配
        """
        dout = self.affine.backward(dscore)
        batch_num, times, hidden2_size = dout.shape
        hidden_size = hidden2_size // 2

        dcontext = dout[:, :, :hidden_size]
        ddecoder_hiddens0 = dout[:, :, hidden_size:]
        dencoder_hiddens, ddecoder_hiddens1 = self.attention.backward(dcontext)
        ddecoder_hiddens = ddecoder_hiddens0 + ddecoder_hiddens1
        dout = self.lstm.backward(ddecoder_hiddens)
        dword_hidden = self.lstm.dhidden
        dencoder_hiddens[:, -1] += dword_hidden
        self.embed.backward(dout)

        return dencoder_hiddens

    def generate(self, encoder_hiddens, start_id, sample_size):
        sampled = []
        sample_id = start_id
        hidden = encoder_hiddens[:, -1]
        self.lstm.set_state(hidden)

        for _ in range(sample_size):
            input = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(input)
            decoder_hiddens = self.lstm.forward(out)
            context = self.attention.forward(encoder_hiddens, decoder_hiddens)
            out = np.concatenate((context, decoder_hiddens), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):
    """Attention付きSeq2seqクラス

    Attributes:
        encoder (AttentionEncoder): Attention付きEncoderクラス
        decoder (AttentionDecoder): Attention付きDecoderクラス
        softmax (TimeSoftmaxWithLoss): SoftmaxWithLossレイヤ
        params (list): パラメータを保持するリスト
        grads (list): 勾配を保持するリスト
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """コンストラクタ

        Args:
            vocab_size (int): 語彙数(文字の種類数)
            wordvec_size (int): 単語ベクトルの次元数
            hidden_size (int): 隠れ状態ベクトルの次元数
        """
        self.encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, wordvec_size, hidden_size)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
