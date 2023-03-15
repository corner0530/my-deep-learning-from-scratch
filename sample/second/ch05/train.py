from simple_rnnlm import SimpleRnnlm

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb

if __name__ == "__main__":
    # ハイパーパラメータの設定
    BATCH_SIZE = 10
    WORDVEC_SIZE = 100
    HIDDEN_SIZE = 100  # RNNの隠れ状態ベクトルの要素数
    TIME_SIZE = 5  # Truncated BPTTの展開する時間サイズ
    LEARNING_RATE = 0.1
    MAX_EPOCH = 100

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    corpus_size = 1000  # 先頭の1000個の単語のみ
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    inputs = corpus[:-1]  # 入力
    labels = corpus[1:]  # 教師ラベル

    # モデルの生成
    model = SimpleRnnlm(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE)
    optimizer = SGD(LEARNING_RATE)
    trainer = RnnlmTrainer(model, optimizer)

    trainer.fit(inputs, labels, MAX_EPOCH, BATCH_SIZE, TIME_SIZE)
    trainer.plot()
