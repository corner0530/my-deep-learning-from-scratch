# 『ゼロから作る Deep Learning』シリーズの読書記録

- 『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』
- 『[ゼロから作る Deep Learning ❷ ―自然言語処理編](https://www.oreilly.co.jp/books/9784873118369/)』

## 実装上の注意

### common/layers

- 全てのレイヤはメソッドとして`forward()`と`backward()`を持つ
- 全てのレイヤはインスタンス変数として`params`と`grads`を持つ

### common/optimizer

- 全てのパラメータの更新を行うクラスは`update(params, grads)`を持つ
  - `params`と`grads`の同じインデックスには対応するパラメータと勾配がそれぞれ格納されているとする

### common/np

- GPU を使用するときは`GPU`を`True`に

## 改善点？

- 重みは辞書で格納した方が良い？

## 参考にしたサイトなど

- [『ゼロから作る Deep Learning』のソースコード](https://github.com/oreilly-japan/deep-learning-from-scratch)
- [『ゼロから作る Deep Learning ❷ ―自然言語処理編』のソースコード](https://github.com/oreilly-japan/deep-learning-from-scratch-2)
