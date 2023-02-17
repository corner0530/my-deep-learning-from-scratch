import sys

sys.path.append(".")
from simple_cbow import SimpleCBOW

from common.optimizer import Adam
from common.trainer import Trainer
from common.util import convert_one_hot, create_contexts_target, preprocess

if __name__ == "__main__":
    WINDOW_SIZE = 1
    HIDDEN_SIZE = 5
    BATCH_SIZE = 3
    MAX_EPOCH = 1000

    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)

    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, WINDOW_SIZE)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)

    model = SimpleCBOW(vocab_size, HIDDEN_SIZE)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    trainer.fit(contexts, target, max_epoch=MAX_EPOCH, batch_size=BATCH_SIZE)
    trainer.plot()

    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
