from rnnlm_gen import RnnlmGen

from dataset import ptb

if __name__ == "__main__":
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)
    corpus_size = len(corpus)

    model = RnnlmGen()
    model.load_params("./Rnnlm.pkl")

    # start文字とskip文字の設定
    start_word = "you"
    start_id = word_to_id[start_word]
    skip_words = ["N", "<unk>", "$"]
    skip_ids = [word_to_id[word] for word in skip_words]

    # 文章生成
    word_ids = model.generate(start_id, skip_ids)
    text = " ".join([id_to_word[id] for id in word_ids])  # `"区切り文字".join(リスト)` で単語を連結
    text = text.replace(" <eos>", ".\n")
    print(text)
