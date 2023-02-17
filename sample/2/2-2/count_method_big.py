# coding: utf-8

import sys

sys.path.append(".")

import numpy as np

from common.util import create_co_matrix, most_similar, ppmi
from dataset import ptb

if __name__ == "__main__":
    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)
    print("counting co-occurrence ...")
    co_matrix = create_co_matrix(corpus, vocab_size, window_size)
    print("calculating PPMI ...")
    ppmi_matrix = ppmi(co_matrix, verbose=True)

    print("calculating SVD ...")
    try:
        # truncated SVD
        from sklearn.utils.extmath import randomized_svd

        U, S, V = randomized_svd(
            ppmi_matrix, n_components=wordvec_size, n_iter=5, random_state=None
        )  # 乱数を使ったTruncated SVDで，特異値の大きい物だけに限定して計算する(高速化)
    except ImportError:
        # SVD
        U, S, V = np.linalg.svd(ppmi_matrix)

    word_vecs = U[:, :wordvec_size]

    querys = ["you", "year", "car", "toyota"]
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
