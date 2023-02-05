# coding: utf-8
import sys

sys.path.append(".")  # 親ディレクトリのファイルをインポートするための設定
from simple_convnet import SimpleConvNet

from common.optimizer import *
from common.trainer import Trainer
from dataset import mnist

# ハイパーパラメータの設定
max_epoch = 20
batch_size = 100
hidden_size = 100

# データの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(flatten=False)
model = SimpleConvNet(input_dim=(1, 28, 28), hidden_size=100, output_size=10, weight_init_std=0.01)
optimizer = Adam()

trainer = Trainer(model, optimizer)
trainer.fit(x_train, t_train, max_epoch, batch_size, eval_interval=100)
trainer.plot()
