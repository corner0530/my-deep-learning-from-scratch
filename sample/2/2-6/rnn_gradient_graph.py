import matplotlib.pyplot as plt
import numpy as np

# from common.util import clip_grads

if __name__ == "__main__":
    N = 2
    H = 3
    T = 20

    dh = np.ones((N, H))
    np.random.seed(3)
    Wh = np.random.randn(H, H)
    # Wh = np.random.randn(H, H) * 0.5

    norm_list = []
    for t in range(T):
        dh = np.dot(dh, Wh.T)
        # clip_grads(dh, 5.0)
        norm = np.sqrt(np.sum(dh * dh)) / N
        norm_list.append(norm)

    plt.plot(np.arange(len(norm_list)), norm_list)
    plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    plt.xlabel("time step")
    plt.ylabel("norm")
    plt.show()
