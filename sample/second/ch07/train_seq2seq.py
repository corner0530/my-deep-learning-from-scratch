import matplotlib.pyplot as plt
import numpy as np
from peeky_seq2seq import PeekySeq2seq

from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from dataset import sequence

# from seq2seq import Seq2seq


if __name__ == "__main__":
    # データセットの読み込み
    (x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
    char_to_id, id_to_char = sequence.get_vocab()

    # 反転
    IS_REVERSE = True
    if IS_REVERSE:
        x_train = x_train[:, ::-1]
        x_test = x_test[:, ::-1]

    # ハイパーパラメータの設定
    vocab_size = len(char_to_id)
    WORDVEC_SIZE = 16
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    MAX_EPOCH = 25
    MAX_GRAD = 5.0

    # モデル・オプティマイザ・トレーナの生成
    model = PeekySeq2seq(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE)
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
            verbose = i < 10  # 最初の10個だけ表示
            correct_num += eval_seq2seq(
                model, question, correct, id_to_char, verbose, IS_REVERSE
            )
        acc = float(correct_num) / len(x_test)  # 正解率
        acc_list.append(acc)
        print("val acc %.3f%%" % (acc * 100))

    # グラフの描画
    x = np.arange(len(acc_list))
    plt.plot(x, acc_list, marker="o")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.show()
