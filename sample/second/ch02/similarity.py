from common.util import cos_similarity, create_co_matrix, preprocess

if __name__ == "__main__":
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[word_to_id["you"]]  # 「you」の単語ベクトル
    c1 = C[word_to_id["i"]]  # 「i」の単語ベクトル
    print(cos_similarity(c0, c1))
