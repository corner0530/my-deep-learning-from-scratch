import matplotlib.pyplot as plt
import numpy as np

from common.util import create_co_matrix, ppmi, preprocess

if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(id_to_word)
    co_matrix = create_co_matrix(corpus, vocab_size, window_size=1)
    ppmi_matrix = ppmi(co_matrix)

    U, S, V = np.linalg.svd(ppmi_matrix)  # SVD

    np.set_printoptions(precision=3)  # 有効桁3桁で表示
    print(co_matrix[0])  # 共起行列
    print(ppmi_matrix[0])  # PPMI
    print(U[0])  # SVDを実行した後の単語ベクトル

    # 各単語を2次元のベクトルで表してプロット
    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))  # 単語の文字列をプロット
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.show()
