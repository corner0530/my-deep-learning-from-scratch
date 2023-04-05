# 『ゼロから作る Deep Learning』シリーズの読書記録

- 『[ゼロから作る Deep Learning](http://www.oreilly.co.jp/books/9784873117584/)』
- 『[ゼロから作る Deep Learning ❷ ―自然言語処理編](https://www.oreilly.co.jp/books/9784873118369/)』
- 『[ゼロから作る Deep Learning ❸ ―フレームワーク編](https://www.oreilly.co.jp/books/9784873119069/)』については[my-DeZero](https://github.com/corner0530/my-DeZero)で管理

## 実装上の注意

### 全般

このディレクトリのファイルを読み込むため，最初に以下を実行してインストールする

- conda の場合
  ```bash
  conda develop .
  ```
- pip の場合
  ```bash
  pip install -e .
  ```

アンインストールする場合は

- conda の場合
  ```bash
  conda develop -u .
  ```
- pip の場合
  ```bash
  pip uninstall .
  ```

### common/layers

- 全てのレイヤはメソッドとして`forward()`と`backward()`を持つ
- 全てのレイヤはインスタンス変数として`params`と`grads`を持つ

### common/optimizer

- 全てのパラメータの更新を行うクラスは`update(params, grads)`を持つ
  - `params`と`grads`の同じインデックスには対応するパラメータと勾配がそれぞれ格納されているとする

### common/np

- GPU を使用するときは`GPU`を`True`に

## 参考にしたサイトなど

### 本家

- [『ゼロから作る Deep Learning』のソースコード](https://github.com/oreilly-japan/deep-learning-from-scratch)
- [『ゼロから作る Deep Learning ❷ ―自然言語処理編』のソースコード](https://github.com/oreilly-japan/deep-learning-from-scratch-2)

### モジュール関連

- [sys.path.append() を使わないでください - Qiita](https://qiita.com/siida36/items/b171922546e65b868679)
- [How to install my own python module (package) via conda and watch its changes - Stack Overflow](https://stackoverflow.com/questions/49474575/how-to-install-my-own-python-module-package-via-conda-and-watch-its-changes)
