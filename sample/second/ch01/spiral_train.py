from two_layer_net import TwoLayerNet

from common.optimizer import SGD
from common.trainer import Trainer
from dataset import spiral

if __name__ == "__main__":
    # ハイパーパラメータの設定
    MAX_EPOCH = 300
    BATCH_SIZE = 30
    HIDDEN_SIZE = 10
    LEARNING_RATE = 0.1

    x, t = spiral.load_data()
    model = TwoLayerNet(
        input_size=x.shape[1], hidden_size=HIDDEN_SIZE, output_size=t.shape[1]
    )
    optimizer = SGD(lr=LEARNING_RATE)

    trainer = Trainer(model, optimizer)
    trainer.fit(x, t, MAX_EPOCH, BATCH_SIZE, eval_interval=10)
    trainer.plot()
