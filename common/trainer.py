# coding: utf-8
"""学習を行うクラス

Attributes:
    Trainer (class): 学習を行うクラス
"""
import sys

sys.path.append(".")  # 親ディレクトリのファイルをインポートするための設定
import time

import matplotlib.pyplot as plt
import numpy

from common.np import *  # import numpy as np

# from common.util import clip_grads


class Trainer:
    """学習を行うクラス

    Attributes:
        model (class): モデル
        optimizer (class): 最適化手法
        loss_list (list): 損失関数の値を格納するリスト
        eval_interval (int): 検証を行う間隔
        current_epoch (int): 現在のエポック数
    """

    def __init__(self, model, optimizer):
        """コンストラクタ

        Args:
            model (class): モデル
            optimizer (class): 最適化手法
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(
        self, data, labels, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20
    ):
        """学習を行う

        Args:
            data (ndarray): 入力データ
            labels (ndarray): 教師ラベル
            max_epoch (int, optional): エポック数
            batch_size (int, optional): バッチサイズ
            max_grad (float, optional): 勾配の最大ノルム
            eval_interval (int, optional): 結果を表示する間隔
        """
        data_size = len(data)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # データのシャッフル
            idx = np.random.permutation(data_size)  # データのシャッフル(ランダムな並びを作成する)
            data = data[idx]
            labels = labels[idx]

            for iters in range(max_iters):
                # ミニバッチの取得
                batch_data = data[iters * batch_size : (iters + 1) * batch_size]
                batch_labels = labels[iters * batch_size : (iters + 1) * batch_size]

                # 勾配の計算
                loss = model.forward(batch_data, batch_labels)
                model.backward()
                params, grads = model.params, model.grads
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                # if max_grad is not None:
                # clip_grads(model.grads, max_grad)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # 定期的に学習経過を出力
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        "| epoch %d | iter %d / %d | time %d[s] | loss %.2f"
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            avg_loss,
                        )
                    )
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        """損失関数の値の推移をグラフに描画する

        Args:
            ylim (tuple, optional): y軸の範囲
        """
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("loss")
        plt.show()


def remove_duplicate(params, grads):
    """重み・勾配の重複を削除する

    Args:
        params (list): 重み
        grads (list): 勾配

    Returns:
        list: 重み
        list: 勾配
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        params_len = len(params)

        for i in range(0, params_len - 1):
            for j in range(i + 1, params_len):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配を加算する
                    find_flg = True
                    params.pop(j)  # 重複していた重み・勾配を削除する
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif (
                    params[i].ndim == 2
                    and params[j].ndim == 2
                    and params[i].T.shape == params[j].shape
                    and np.all(params[i].T == params[j])
                ):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads
