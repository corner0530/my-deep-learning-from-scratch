import matplotlib.pyplot as plt
import numpy as np
from attention_seq2seq import AttentionSeq2seq

from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from dataset import sequence

# from sample.second.ch07.peeky_seq2seq import PeekySeq2seq
# from sample.second.ch07.seq2seq import Seq2seq

if __name__ == "__main__":
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = sequence.load_data("date.txt")
    char_to_id, id_to_char = sequence.get_vocab()

    # 入力文を反転
    x_train = x_train[:, ::-1]
    x_test = x_test[:, ::-1]

    # ハイパーパラメータの設定
    vocab_size = len(char_to_id)
    WORDVEC_SIZE = 16
    HIDDEN_SIZE = 256
    BATCH_SIZE = 128
    MAX_EPOCH = 10
    MAX_GRAD = 5.0

    model = AttentionSeq2seq(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE)
    # model = Seq2seq(vocab_size, wordvec_size, hidden_size)
    # model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(MAX_EPOCH):
        trainer.fit(
            x_train, t_train, max_epoch=1, batch_size=BATCH_SIZE, max_grad=MAX_GRAD
        )

        correct_num = 0
        for i in range(len(x_test)):
            question = x_test[[i]]
            correct = t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(
                model, question, correct, id_to_char, verbose, is_reverse=True
            )

        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print("val acc %.3f%%" % (acc * 100))

    model.save_params()

    # グラフの描画
    x = np.arange(len(acc_list))
    plt.plot(x, acc_list, marker="o")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(-0.05, 1.05)
    plt.show()
