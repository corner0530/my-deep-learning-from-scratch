from rnnlm import Rnnlm

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb

if __name__ == "__main__":
    # ハイパーパラメータの設定
    BATCH_SIZE = 20
    WORDVEC_SIZE = 100
    HIDDEN_SIZE = 100  # RNNの隠れ状態ベクトルの要素数
    TIME_SIZE = 35  # RNNを展開するサイズ
    LEARNING_RATE = 20.0
    MAX_EPOCH = 4
    MAX_GRAD = 0.25

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    corpus_test, _, _ = ptb.load_data("test")
    vocab_size = len(word_to_id)
    inputs = corpus[:-1]
    labels = corpus[1:]

    # モデルの生成
    model = Rnnlm(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE)
    optimizer = SGD(LEARNING_RATE)
    trainer = RnnlmTrainer(model, optimizer)

    # 1. 勾配クリッピングを適用して学習
    trainer.fit(
        inputs, labels, MAX_EPOCH, BATCH_SIZE, TIME_SIZE, MAX_GRAD, eval_interval=20
    )
    trainer.plot(ylim=(0, 500))

    # 2. テストデータで評価
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print("test perplexity: ", ppl_test)

    # 3. パラメータの保存
    model.save_params()
