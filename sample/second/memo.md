# 自然言語と単語の分散表現

単語の意味をとらえた表現方法に以下の 3 つがある

- シソーラスによる手法
- カウントベースによる手法
- 推論ベースの手法

## シソーラス

- 類語辞書．同じ意味の単語(同義語)や意味の似た単語(類義語)を同じグループに分類した物．
- 単語の意味を人手で定義している
- 単語の間で，「上位と下位」・「全体と部分」などの細かい関連性が定義されている場合がある

### WordNet

- 最も有名なシソーラス
- 問題点:
  - 時代の変化に対応するのが困難
  - 人の作業コストが高い
  - 単語の細かなニュアンスを表現できない

## カウントベース

- **コーパス**: 大量のテキストデータ
- コーパスから，自動的に効率よく，文書の書き方・単語の選び方・意味などを中質する

### 単語の分散表現

- 単語の意味をベクトルで表現したもの

### 分布仮説

- **単語の意味は，周囲の単語によって形成される**
- コンテキスト: (注目する単語に対して)その周囲に存在する単語
  - コンテキストのサイズをウィンドウサイズという

### 共起行列

- 統計的手法: ある単語に着目した場合，その周囲にどのような単語がどれだけ現れるのかをカウントし，集計する方法
- **共起行列**: 全ての単語に対して，共起する(コンテキストに含まれる)単語をテーブルにまとめたもの
  - 各行が該当する単語のベクトルに対応する

### ベクトル間の類似度

- 単語のベクトル表現の類似度に関しては**コサイン類似度**が良く用いられる．
- コサイン類似度は$\mathbf{x}=\left(x_1,x_2,x_3,\ldots,x_n\right),\mathbf{y}=\left(y_1,y_2,y_3,\ldots,y_n\right)$について，
  $$
  \text{similarity}\left(\mathbf{x},\mathbf{y}\right)=\frac{\mathbf{x}\cdot\mathbf{y}}{\left\|\mathbf{x}\right\|\left\|\mathbf{y}\right\|}=\frac{x_1y_1+\cdots+x_ny_n}{\sqrt{x_1^2+\cdots+x_n^2}\sqrt{y_1^2+\cdots+y_n^2}}
  $$
  ただし$\left\|\cdot\right\|$は L2 ノルム．コサイン類似度は，直感的には「2 つのベクトルがどれだけ同じ方向を向いているか」を表す

### 相互情報量(PMI)

- 共起した回数だけを見るのでは良くないため，相互情報量を用いる．$x,y$を確率変数とし，この値が高いほど関連性が高い．
  $$
  \text{PMI}\left(x,y\right)=\log_2\frac{P\left(x,y\right)}{P\left(x\right)P\left(y\right)}
  $$
  $P(x)$は単語$x$が出現する確率，$P(x,y)$は$x$と$y$が共起する確率．
  ここで単語$x$が出現する回数を$C(x)$，$x$と$y$が共起する回数を$C(x,y)$，コーパスの単語数を$N$として，
  $$
  \text{PMI}\left(x,y\right)=\log_2\frac{C\left(x,y\right)\cdot N}{C\left(x\right)C\left(y\right)}
  $$
  ただし，共起する回数が 0 回だと PMI が負の無限大になるため，実際上は**正の相互情報量**(PPMI)を用いる．
  $$
  \text{PPMI}\left(x,y\right)=\max\left(0,\text{PMI}\left(x,y\right)\right)
  $$

### 次元削減

- ベクトルの次元を「重要な情報」をできるだけ残したうえで削減する手法
  - 疎な行列はより少ない次元で表現しなおせる
- **特異値分解**(SVD): 任意の行列$X$を直交行列$U,V$と対角行列$S$を用いて以下のように分解する
  $$
  X=USV^\top
  $$
  - $U$は直交行列なので何らかの空間の基底を形成している．これを「単語空間」として扱える．
  - $S$は対角行列で，対角成分に「特異値」(「対応する軸」の重要度)が大きい順に並んでいる．→ 重要でない軸を削れる
  - $S$の特異値が小さい部分に相当する列ベクトル(単語ベクトル)を$U$から削ることで元の行列を次元削減できる．
  - 密な行列になるのでノイズに対するロバスト性を高められる

### PTB データセット

- **Penn Treeback**: 手ごろに使えるコーパス
  - 出現の頻度の低い単語を特殊文字で置き換えたり，具体的な数字を`N`で置き換えたりなどしたもの

# word2vec

## one-hot 表現

- ベクトルの要素の中で 1 つだけが 1 で残りがすべて 0 であるようなベクトル → これを入力層とする
  - これに重みを掛けることは重みの行ベクトル(=分散表現)を「抜き出す」ことに相当する

## CBOW(continuous bag-of-words)

- コンテキストを入力し，ターゲットを推測することを目的とした NN
  - 入力層: コンテキスト(one-hot 表現)を単語数だけ入力層として別々に用意する
  - 中間層: それらに共通の重みで全結合層に入力し平均を取る
  - 出力層: 中間層とは別の重みで全結合層に入力し出力をスコアとする(Softmax 関数を適用することで確率とできる)
  - 入力層 → 中間層の全結合層の重み(の各行)が(各)単語の分散表現となる → 意味がエンコードされる
    - 中間層 → 出力層の全結合層の重み(の各列)にも(各)単語の分散表現が格納されるが，こちらはあまりに使われない
    - 全結合層は MatMul レイヤ (バイアスがないため)
  - 学習: Softmax とクロスエントロピー誤差を用いるだけ

## 学習データ

- コーパスからコンテキストとターゲットを作成する

## 補足

### CBOW モデルと確率

$w_1,w_2,\ldots,w_T$という単語列について，ウィンドウサイズが 1 のときにコンテキストとして$w_{t-1}$と$w_{t+1}$が与えられたときのターゲットが$w_t$の確率は$P(w_t\mid w_{t-1},w_{t+1})$で，これをモデル化している．

1 つのサンプルデータに関数の損失関数(**負の対数尤度**)は，

$$
L=-\log P(w_t\mid w_{t-1},w_{t+1})
$$

で，これをコーパス全体に拡張すると，

$$
L=-\frac{1}{T}\sum_{t-1}^T\log P(w_t\mid w_{t-1},w_{t+1})
$$

となり，これを損失関数とする．

### skip-gram モデル

- CBOW で扱うコンテキストとターゲットを逆転させたモデル(中央の単語から周囲の複数ある単語を推測する)
  - $P(w_{t-1},w_{t+1}\mid w_t)$をモデル化したもの
- ネットワークは，入力層が 1 つ，出力層がコンテキストの数だけ存在する．
- $P(w_{t-1},w_{t+1}\mid w_t)=P(w_{t-1}\mid w_t)P(w_{t+1}\mid w_t)$の条件付独立(コンテキストの単語間に関連性がないこと)を仮定して，1 つのデータに対する負の対数尤度は，

  $$
  L=-\left(\log P\left(w_{t-1}\mid w_t\right)+\log P\left(w_{t+1}\mid w_t\right)\right)
  $$

  となり，コーパス全体の損失関数は

  $$
  L=-\frac{1}{T}\sum_{t=1}^T\left(\log P\left(w_{t-1}\mid w_t\right)+\log P\left(w_{t+1}\mid w_t\right)\right)
  $$

- skip-gram モデルは CBOW モデルに比べて，単語の分散表現の精度が優れているが学習速度は遅い．

### カウントベース vs. 推論ベース

- 新しい単語の追加による単語の分散表現の更新の場合，カウントベースは共起行列の作成から行うが，推論ベースでは再学習を行えるので，推論ベースの方が効率的．
- 単語の分散表現の性質や精度については，カウントベースでは単語の類似性が，word2vec ではそれに加えて単語間のパターンをとらえられる．
- しかし優劣はつけられない
- また，skip-gram と Negative Sampling を用いたモデルは，コーパス全体の共起行列に対して特別な行列分解をしているのと同じ
- さらにカウントベースと推論ベースを合わせた GloVe もある

# word2vec の高速化

- CBOW モデルにおいて，語彙数が増えると以下の計算に多くの時間を要する
  - 入力層の one-hot 表現と重み行列$W_{in}$の積による計算(Embedding レイヤで解決)
  - 中間層と重み行列$W_{out}$の積及び Softmax レイヤの計算(Negative Sampling で解決)

## Embedding レイヤ

- 重みパラメータから「単語 ID に該当する行(ベクトル)」を抜き出すレイヤ
  - 単語の埋め込みが格納される

## Negative Sampling

- 「多値分類」を「二値分類」に近似する
  - コンテキストから求めた単語がターゲットと一致するかどうか
- 中間層と出力層の重み行列の積 = 出力層の重み行列からターゲットに対応する列(単語ベクトル)を抽出して中間層のニューロンと内積をとる
  - これをスコアとしてシグモイド関数を適用
- 正例については以上のように学習する．同時に負例についてはいくつかサンプリングしてそれについても損失を求め，それらについての損失を足し合わせて最終的な損失とする．
- 負例のサンプリングは，コーパス中でよく使われる単語をより多く抽出する．
  - コーパス中の各単語の出現回数を以下の確率分布で表す($P\left(w_i\right)$は$i$番目の単語の確率)．
    $$
    P^\prime\left(w_i\right)=\frac{P\left(w_i\right)^{0.75}}{\sum_j^nP\left(w_j\right)^{0.75}}
    $$

## 補足

### アプリケーション

- **転移学習**: ある分野で学習したデータを別の分野にも適用できる
  - 大きなコーパスで先に学習し，その分散表現を個別のタスクで利用する
- 1 つの単語を固定長のベクトルに変換できる
  - 文章に対しても同様に変換でき，最も単純な方法として bag-of-words がある
    - 各単語を分散表現に変換し，それらの総和を求める
    - RNN を用いてもできる
  - 単語や文章を固定長のベクトルに変換することで，NN や SVM など一般的な機械学習の手法に適用できる

### 単語ベクトルの評価方法

- 現実的なアプリケーションでは，単語の分散表現の学習と，特定の問題について分類を行うシステムの学習の 2 段階の学習を行うことが多く，チューニングに多くの時間がかかる．
- そのため，単語の「類似性」や「類推問題」で評価することが一般的．
  - 人が作成した単語類似度の評価セットを用いて比較
  - 類推問題による評価でもその正解率を用いる
    - モデルによって精度が異なる(コーパスに応じて最適なモデルを選ぶ)
    - コーパスが大きいほど良い結果になる(ビッグデータは常に望まれる)
    - 単語ベクトルの次元数は適度な大きさが必要(大きすぎても精度が悪くなる)

# リカレントニューラルネットワーク(RNN)

- **フィードフォワード**は流れが一方向のネットワークで，時系列データをうまく扱えないため，RNN を用いる
- **言語モデル**: 単語の並びに対して，それがどれだけ起こりえるのかを確率で評価するモデル
  $$
  P\left(w_1,\ldots,w_m\right)=\prod_{t=1}^mP\left(w_t\mid w_1,\ldots,w_{t-1}\right)
  $$
  - ここで右辺の事後確率は，対象の単語より左の全ての単語をコンテキストとしたときの確率の総乗
- RNN はループする経路を持ちデータが循環する
  - 時刻を$t$として$x_t$を入力とし，$h_t$が出力される
- ループを展開すると，フィードフォワード型と同じ形ですべて同じレイヤにあることになる
- 入力$x$を出力$h$に変換する重み$W_x$と 1 つ前の RNN の出力を次の時刻の出力に変換する重み$W_h$，バイアス$b$を用いて以下のように表せる
  $$
  h_t=\tanh\left(h_{t-1}W_h+x_tW_x+b\right)
  $$
  - なお，$h_t$を**隠れ状態**(**隠れ状態ベクトル**)と呼ばれる
- **Backpropagation Through Time(BPTT)**: 時間方向に展開したニューラルネットワークの誤差逆伝播法
  - 時系列データが長くなると計算リソースが増加する
- **Truncated BPTT**: ネットワークのつながり(逆伝播のみ)を適当な長さで断ち切ってネットワークを作る BPTT
  - データをシーケンシャルに与える
  - ミニバッチ学習を行うには，(サンプル数)/(バッチ数)個だけオフセットを取る
- **パープレキシティ**: 言語モデルの予測性能の良さを評価する指標として良く用いられる「確率の逆数」=平均分岐数で，データの個数を$N$個，$\mathbf{t}_n$を one-hot vector の正解ラベル，$t_{nk}$を$n$個目のデータの$k$番目の値，$y_{nk}$を確率分布として，以下で表される
  $$
  \begin{align*}
    L&=-\frac{1}{N}\sum_n\sum_kt_{nk}\log y_{nk}\\
    \text{perplexity}&=e^L
  \end{align*}
  $$

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

## 付録より

- **GRU**: ゲート付き RNN の 1 つ．記憶セルを削減し，計算時間を短縮している
  - **update ゲート**: 隠れ状態を更新するゲート
    $$
    \bm{z}=\sigma\left(\bm{x}_t\bm{W_x^{\left(z\right)}}+\bm{h}_{t-1}\bm{W_h^{\left(z\right)}}+\bm{b^{\left(z\right)}}\right)
    $$
  - **reset ゲート**: 過去の隠れ状態をどれだけ無視するかを決定する
    $$
    \bm{r}=\sigma\left(\bm{x}_t\bm{W_x^{\left(r\right)}}+\bm{h}_{t-1}\bm{W_h^{\left(r\right)}}+\bm{b^{\left(r\right)}}\right)
    $$
  - 新しい隠れ状態
    $$
    \bm{\tilde{h}}=\tanh\left(\bm{x}_t\bm{W_x}+\left(\bm{r}\odot\bm{h}_{t-1}\right)\bm{W_h}+\bm{b}\right)
    $$
  - 隠れ状態
    $$
    \bm{h}_t=\left(1-\bm{z}\right)\odot\bm{h}_{t-1}+\bm{z}\odot\bm{\tilde{h}}
    $$

# RNN による文章生成

- **seq2seq**: 時系列データを別の時系列データに変換するモデルの 1 つで 2 つの RNN を利用する
  - Encoder-Decoder モデル(Encoder でエンコードした情報から目的とする文章を Decoder で生成する)
    - Encoder は Embedding レイヤ →LSTM レイヤを用いて隠れ状態ベクトル$\bm{h}$に変換する
    - Decoder は LSTM レイヤで$\bm{h}$を受け取って文章生成を行う
- トイ・プロブレム: 機械学習を評価するために作られた簡単な問題
- 可変長の時系列データを扱う →**パディング**を行う(無効なデータで埋めてデータの長さを均一にする)
  - 無効なデータを処理するため，パディング専用の処理を seq2seq に加える必要がある
    - Decoder でパディングが入力されたときには損失の結果に計上しないようにする ←Softmax with Loss レイヤに「マスク」を追加する
    - Encoder でパティングが入力されたときには前時刻の入力をそのまま出力
- 改良
  - 入力データの反転(Reverse): 学習の進みが早くなり，精度もよくなる(勾配の伝播がスムーズになる(反転すると対応関係にある返還後の単語との距離が近くなることが多い)ため)
  - **覗き見(Peeky)**: Encoder の出力$\bm{h}$を Decoder の他のレイヤにも与える
- 応用
  - 機械翻訳・自動要約・質疑応答・メールの自動返信
  - アルゴリズムの学習
  - **イメージキャプション**: 画像を文章へ変換する(Encoder を CNN に置き換える)

# Attention

- **Attention(注意機構)**: 必要な情報に注意を向けさせる
- seq2seq では時系列データを固定長のベクトルに変換するが，これを改良する
- Encoder では各時刻の LSTM レイヤの隠れ状態ベクトルを全て利用する(ベクトルの長さは時刻により変わる)
  - 各時刻の隠れ状態にはその時刻で入力された単語の情報が多く含まれる
- Decoder では，Encoder の隠れ状態ベクトル全てを活用するため，以下の改良を行う
  1. 入力と出力でどの単語が関連しているかの対応関係(**アライメント**)を学習させるため，対応関係にある元の単語の情報を選び出すように注意を向けさせる
  - ただし実際は選び出すのではなく全ての単語について重みづけし，この重みづけ和を足し合わせてコンテキストベクトルとする
  2. 1 の重みの学習は以下のように行う
  - 各単語について，LSTM レイヤの隠れ状態ベクトルと，Encoder の各単語の隠れ状態ベクトルの類似度を内積で算出しスコアとする
  3. Encoder が出力する各単語のベクトルに対し，2 を用いて注意を払って各単語の重みを求め，この重みを用いて 1 で重みづけ和を求め，これをコンテキストベクトルとする．そしてこのコンテキストベクトルを LSTM レイヤの入力に追加する
- 翻訳用のデータセットとして「WTM」などが有名
- 双方向 LSTM/双方向 RNN:逆方向に処理する LSTM レイヤを追加し，順方向と連結する．これにより両方向からの情報を集約できる
- Attention レイヤの場所は LSTM レイヤと Affine レイヤの間以外に，次の時刻の LSTM レイヤの入力の前などにおくこともある
- seq2seq において LSTM レイヤを多層にすると表現力がより高くなる(但し Encoder と Decoder で層数は同じにするのが一般的)
- **skip コネクション**: 層を深くする際に層をまたいで接続するテクニック．接続部で 2 つの出力が加算される
- 応用
  - **GNMT**(Google Neural Machine Translation): 機械翻訳
  - **Transformer**: RNN は並列的に計算できないため RNN ではなく Attention を使って処理する
    - **Self-Attention**: 1 つの時系列データを対象とした Attention であり，1 つの時系列データ内において各要素が他の要素に対してどのような関連性があるのかを見るもの
  - **NTM**(Neural Turing Machine): ニューラルネットワークを外部メモリを利用
    - Encoder が必要な情報をメモリに書き込み，Decoder はそのメモリにある情報から必要な情報を読み込んでいると解釈すると，コンピュータのメモリ操作を NN で再現できそう
