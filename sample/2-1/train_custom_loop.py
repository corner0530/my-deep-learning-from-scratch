import sys

sys.path.append(".")  # 親ディレクトリのファイルをインポートするための設定
import matplotlib.pyplot as plt
import numpy as np
from two_layer_net import TwoLayerNet

from common.optimizer import SGD
from dataset import spiral

# ハイパーパラメータの設定
MAX_EPOCH = 300
BATCH_SIZE = 30
HIDDEN_SIZE = 10
LEARNING_RATE = 1.0

# データの読み込み、モデルとオプティマイザの生成
data, labels = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=HIDDEN_SIZE, output_size=3)
optimizer = SGD(lr=LEARNING_RATE)

# 学習で使用する変数
data_size = len(data)
max_iters = data_size // BATCH_SIZE
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(MAX_EPOCH):
    # データのシャッフル
    idx = np.random.permutation(data_size)  # データのシャッフル(ランダムな並びを作成する)
    data = data[idx]
    labels = labels[idx]

    for iters in range(max_iters):
        # ミニバッチの取得
        batch_data = data[iters * BATCH_SIZE : (iters + 1) * BATCH_SIZE]
        batch_labels = labels[iters * BATCH_SIZE : (iters + 1) * BATCH_SIZE]

        # 勾配の計算
        loss = model.forward(batch_data, batch_labels)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 定期的に学習経過を出力
        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(
                "| epoch %d | iter %d / %d | loss %.2f"
                % (epoch + 1, iters + 1, max_iters, avg_loss)
            )
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

# 学習結果のプロット
plt.plot(np.arange(len(loss_list)), loss_list, label="train")
plt.xlabel("iterations (x10)")
plt.ylabel("loss")
plt.show()

# 境界領域のプロット
h = 0.001
data_min, data_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
label_min, label_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
xx, yy = np.meshgrid(
    np.arange(data_min, data_max, h), np.arange(label_min, label_max, h)
)  # 格子点
X = np.c_[xx.ravel(), yy.ravel()]  # xxとyyを横方向に結合
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)  # 等高線のプロット
plt.axis("off")

# データ点のプロット
data, labels = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ["o", "x", "^"]
for i in range(CLS_NUM):
    plt.scatter(
        data[i * N : (i + 1) * N, 0],
        data[i * N : (i + 1) * N, 1],
        s=40,
        marker=markers[i],
    )
plt.show()
