import numpy as np

from common.util import clip_grads

if __name__ == "__main__":
    dW1 = np.random.randn(3, 3) * 10
    dW2 = np.random.randn(3, 3) * 10
    grads = [dW1, dW2]
    max_norm = 5.0

    print('before:', dW1.flatten())
    clip_grads(grads, max_norm)
    print('after:', dW1.flatten())
