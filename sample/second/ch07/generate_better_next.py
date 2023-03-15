import numpy as np
from rnnlm_gen import BetterRnnlmGen

from dataset import ptb

if __name__ == "__main__":
    corpus, word_to_id, id_to_word = ptb.load_data("train")
    vocab_size = len(word_to_id)
    corpus_size = len(corpus)

    model = BetterRnnlmGen()
    model.load_params("./BetterRnnlm.pkl")

    # "you"で始める
    start_word = "you"
    start_id = word_to_id[start_word]
    skip_words = ["N", "<unk>", "$"]
    skip_ids = [word_to_id[word] for word in skip_words]
    word_ids = model.generate(start_id, skip_ids)
    text = " ".join([id_to_word[id] for id in word_ids])
    text = text.replace(" <eos>", ".\n")
    print(text)
    print('-' * 50)

    # "the meaning of life is"で始める
    model.reset_state()
    start_words = 'the meaning of life is'
    start_ids = [word_to_id[word] for word in start_words.split(' ')]
    for x in start_ids[:-1]:
        x = np.array(x).reshape(1, 1)
        model.predict(x)
    word_ids = model.generate(start_ids[-1], skip_ids)
    word_ids = start_ids[:-1] + word_ids
    text = ' '.join([id_to_word[id] for id in word_ids])
    text = text.replace(' <eos>', '.\n')
    print(text)
