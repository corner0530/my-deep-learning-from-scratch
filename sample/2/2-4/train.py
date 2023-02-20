import pickle

from cbow import CBOW

from common.np import GPU, np  # import numpy as np
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

if __name__ == "__main__":
    # ハイパーパラメータの設定
    WINDOW_SIZE = 5
    HIDDEN_SIZE = 100
    BATCH_SIZE = 100
    MAX_EPOCH = 10

    # データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, WINDOW_SIZE)
    if GPU:
        contexts = to_gpu(contexts)
        target = to_gpu(target)

    # モデルなどの生成
    model = CBOW(vocab_size, HIDDEN_SIZE, WINDOW_SIZE, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # 学習
    trainer.fit(contexts, target, MAX_EPOCH, BATCH_SIZE)
    trainer.plot()

    # データを保存
    word_vecs = model.word_vecs
    if GPU:
        word_vecs = to_cpu(word_vecs)

    params = {
        "word_vecs": word_vecs.astype(np.float16),
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
    }
    pkl_file = "cbow_params.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(params, f, -1)
