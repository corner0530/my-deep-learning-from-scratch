# coding: utf-8
"""スパイラル・データセット

Attributes:
    load_data (function): スパイラル・データセットを読み込む
"""
import numpy as np


def load_data(seed=1984):
    """スパイラル・データセットを読み込む

    Args:
        seed (int): 乱数のシード

    Returns:
        ndarray: 入力データ
        ndarray: 教師データ
    """
    np.random.seed(seed)
    N = 100  # クラスごとのサンプル数
    DIM = 2  # データの要素数
    CLS_NUM = 3  # クラス数

    x = np.zeros((N * CLS_NUM, DIM))  # 入力データ
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int32)  # 教師データ

    for j in range(CLS_NUM):
        for i in range(N):  # (N*j, N*(j+1)):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t
