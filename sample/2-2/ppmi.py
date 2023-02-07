import sys

sys.path.append(".")
import numpy as np

from common.util import create_co_matrix, ppmi, preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
co_matrix = create_co_matrix(corpus, vocab_size)
ppmi_matrix = ppmi(co_matrix)

np.set_printoptions(precision=3)  # 有効桁3桁で表示
print('covariance matrix')
print(co_matrix)
print('-'*50)
print('PPMI')
print(ppmi_matrix)
