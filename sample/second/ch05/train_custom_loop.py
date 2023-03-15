import matplotlib.pyplot as plt
import numpy as np
from simple_rnnlm import SimpleRnnlm

from common.optimizer import SGD
from dataset import ptb

if __name__ == "__main__":
    # ハイパーパラメータの設定
    BATCH_SIZE = 10
    WORDVEC_SIZE = 100
    HIDDEN_SIZE = 100  # RNNの隠れ状態ベクトルの要素数
    TIME_SIZE = 5  # Truncated BPTTの展開する時間サイズ
    LEARNING_RATE = 0.1
    MAX_EPOCH = 100

    # 学習データの読み込み・データセットを小さくする
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    corpus_size = 1000  # 先頭の1000個の単語のみ
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    inputs = corpus[:-1]  # 入力
    labels = corpus[1:]  # 教師ラベル
    data_size = len(inputs)
    print("corpus size: %d, vocabulary size: %d" % (corpus_size, vocab_size))

    # 学習時に使用する変数
    max_iters = data_size // (BATCH_SIZE * TIME_SIZE)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []

    # モデルの生成
    model = SimpleRnnlm(vocab_size, WORDVEC_SIZE, HIDDEN_SIZE)
    optimizer = SGD(LEARNING_RATE)
    # optimizer = Adam()

    # 1. ミニバッチの各サンプルの読み込み開始位置を計算して格納
    jump = (corpus_size - 1) // BATCH_SIZE
    offsets = [i * jump for i in range(BATCH_SIZE)]

    for epoch in range(MAX_EPOCH):
        for iter in range(max_iters):
            # 2. ミニバッチの取得
            batch_input = np.empty((BATCH_SIZE, TIME_SIZE), dtype="i")
            batch_label = np.empty((BATCH_SIZE, TIME_SIZE), dtype="i")
            for time in range(TIME_SIZE):
                for i, offset in enumerate(offsets):
                    # ミニバッチの各サンプルの読み込み開始位置をずらしながら単語IDを取得
                    batch_input[i, time] = inputs[(offset + time_idx) % data_size]
                    batch_label[i, time] = labels[(offset + time_idx) % data_size]
                time_idx += 1

            # 勾配を求め，パラメータを更新
            loss = model.forward(batch_input, batch_label)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # 3. エポックごとにパープレキシティの評価
        ppl = np.exp(total_loss / loss_count)
        print("| epoch %d | perplexity %.2f" % (epoch + 1, ppl))
        ppl_list.append(float(ppl))
        total_loss = 0
        loss_count = 0

    # グラフの描画
    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label="train")
    plt.xlabel("epochs")
    plt.ylabel("perplexity")
    plt.show()
