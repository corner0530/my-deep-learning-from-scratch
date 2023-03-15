# ゲート付き RNN

- シンプルな RNN は BPTT で勾配消失 or 勾配爆発が起きるので長期記憶が苦手
  - $\tanh$については ReLU を代わりに用いてもよい
  - 重みについて特異値(データにどれだけ広がりがあるか)の最大値が 1 より大きいかどうかで勾配のノルムの変化が異なる
- **勾配クリッピング**: 勾配爆発への対策で，全てのパラメータに対する勾配を$\hat{g}$にまとめ，閾値を threshold としたときに，もし$\left\|\hat{g}\right\|\geq\text{threshold}$ならば$\hat{g}=\frac{\text{threshold}}{\left\|\hat{g}\right\|}\hat{g}$とする手法
- **LSTM**: ゲート付き RNN の 1 つ
  - LSTM レイヤ内に**記憶セル**(LSTM 専用の記憶部)があり，これを元に外部のレイヤへ隠れ状態を出力する
    - 記憶セルに過去から時刻$t$迄において必要な情報が格納されている
  - 隠れ状態$\bm{h}_t$は記憶セル$\bm{c}_t$を用いて$\bm{h}_t=\tanh\left(\bm{c}_t\right)$
- ゲート: 開き具合をコントロールする重みパラメータがあり，sigmoid 関数を用いて開き具合を求める
  - **output ゲート**: $\tanh(c_t)$の各要素に対してそれらが隠れ状態としてどれだけ重要か(次へ何%だけ通すか)を調整するゲート
    出力$\bm{o}$は
    $$
    \bm{o}=\sigma\left(\bm{x}_t\bm{W_x^{\left(o\right)}}+\bm{h}_{t-1}\bm{W_h^{\left(o\right)}}+\bm{b^{\left(o\right)}}\right)
    $$
    隠れ状態$\bm{h}_t$は，$\odot$をアダマール積として
    $$
    \bm{h}_t=\bm{o}\odot\tanh\left(\bm{c}_t\right)
    $$
  - **forget ゲート**: $\bm{c}_{t-1}$の記憶から不要な記憶を忘れるためのゲート
    出力$\bm{f}$は
    $$
    \bm{f}=\sigma\left(\bm{x}_t\bm{W_x^{\left(f\right)}}+\bm{h}_{t-1}\bm{W_h^{\left(f\right)}}+\bm{b^{\left(f\right)}}\right)
    $$
  - 新しい記憶セル: 新しく覚える情報を記憶セルに追加する
    新しい記憶$\bm{g}$は
    $$
    \bm{g}=\tanh\left(\bm{x}_t\bm{W_x^{\left(g\right)}}+\bm{h}_{t-1}\bm{W_h^{\left(g\right)}}+\bm{b^{\left(g\right)}}\right)
    $$
  - **input ゲート**: $\bm{g}$の各要素が新たに追加する情報としてどれだけ価値があるかを判断するゲート
    $$
    \bm{i}=\sigma\left(\bm{x}_t\bm{W_x^{\left(i\right)}}+\bm{h}_{t-1}\bm{W_h^{\left(i\right)}}+\bm{b^{\left(i\right)}}\right)
    $$
    記憶セル$\bm{c}_t$は
    $$
    \bm{c}_t=\bm{f}\odot\bm{c}_{t-1}+\bm{g}\odot\bm{i}
    $$
- 改善点
  - LSTM レイヤを多層化する(2~4 程度)
  - **過学習**の抑制
    - Dropout のレイヤを時間方向ではなく深さ方向に挿入する(時間方向に挿入すると，時間が進むのに比例してノイズが蓄積することになるため)
    - 時間方向の正則化を目的としたものとして，変分 Dropout などがある
      - これは同じ階層にある Dropout ではマスクを共有する
  - **重み共有**: Embedding レイヤの重みと Affine レイヤの重みを共有する(ただしサイズが異なるので転置を取る)
