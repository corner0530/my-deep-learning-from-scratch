"""関数の実装

Attributes:
    sigmoid (function): シグモイド関数
    relu (function): ReLU
    softmax (function): ソフトマックス関数
    cross_entropy_error (function): クロスエントロピー誤差
"""
from common.np import np  # import numpy as np


def sigmoid(x):
    """シグモイド関数

    Args:
        x (ndarray): 入力値

    Returns:
        ndarray: 出力値
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU

    Args:
        x (ndarray): 入力値

    Returns:
        ndarray: 出力値
    """
    return np.maximum(0, x)


def softmax(x):
    """ソフトマックス関数

    Args:
        x (ndarray): 入力値

    Returns:
        ndarray: 出力値
    """
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    """クロスエントロピー誤差

    Args:
        y (ndarray): 出力値
        t (ndarray): 正解ラベル

    Returns:
        float: クロスエントロピー誤差の平均
    """
    # 1枚のとき
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
