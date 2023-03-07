from better_rnnlm import BetterRnnlm

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb

if __name__ == "__main__":
    # ハイパーパラメータの設定
    BATCH_SIZE = 20
    WORDVEC_SIZE = 650
    HIDDEN_SIZE = 650
    TIME_SIZE = 35
    MAX_EPOCH = 40
    MAX_GRAD = 0.25
    DROPOUT = 0.5
    learning_rate = 20.0

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    corpus_val, _, _ = ptb.load_data("val")
    corpus_test, _, _ = ptb.load_data("test")

    vocab_size = len(word_to_id)
    inputs = corpus[:-1]
    labels = corpus[1:]

    model = BetterRnnlm(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE, DROPOUT)
    optimizer = SGD(learning_rate)
    trainer = RnnlmTrainer(model, optimizer)

    best_ppl = float("inf")
    for epoch in range(MAX_EPOCH):
        trainer.fit(
            inputs,
            labels,
            max_epoch=1,
            batch_size=BATCH_SIZE,
            time_size=TIME_SIZE,
            max_grad=MAX_GRAD,
        )

        # エポックごとにパープレキシティを評価
        model.reset_state()
        ppl = eval_perplexity(model, corpus_val)
        print("valid perplexity: ", ppl)

        if best_ppl > ppl:
            # パープレキシティが改善した場合のみモデルを保存
            best_ppl = ppl
            model.save_params()
        else:
            # パープレキシティが悪化した場合は学習率を下げる
            learning_rate /= 4.0
            optimizer.lr = learning_rate

        model.reset_state()
        print("-" * 50)

    # テストデータでの評価
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print("test perplexity: ", ppl_test)
