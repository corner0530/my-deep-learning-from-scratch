import numpy as np

from common.layers import MatMul

if __name__ == "__main__":
    # サンプルのコンテキストデータ
    context0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
    context1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

    # 重みの初期化
    weight_in = np.random.randn(7, 3)
    weight_out = np.random.randn(3, 7)

    # レイヤの生成
    in_layer0 = MatMul(weight_in)
    in_layer1 = MatMul(weight_in)
    out_layer = MatMul(weight_out)

    # 順伝播
    h0 = in_layer0.forward(context0)
    h1 = in_layer1.forward(context1)
    h = 0.5 * (h0 + h1)
    s = out_layer.forward(h)

    print(s)
