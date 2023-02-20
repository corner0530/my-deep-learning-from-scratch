"""ユーティリティ関数の実装

Attributes:
    im2col (function): 畳み込み用に画像を2次元配列に変換
    col2im (function): 畳み込み用に勾配を4次元配列に変換
    preprocess (function): コーパスの前処理を行う
    create_co_matrix (function): コーパスから共起行列を作成する
    cos_similarity (function): コサイン類似度を計算する
    most_similar (function): 類似単語のランキング表示
    ppmi (function): PPMI行列を作成する
    create_contexts_target (function): コンテキストとターゲットの作成
    convert_one_hot (function): one-hot表現への変換
"""
from common.np import np  # import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """フィルターサイズ・ストライド・パディングを指定して入力データを2次元配列に変換する

    Args:
        input_data (ndarray): 入力データ(データ数, チャンネル, 高さ, 幅)
        filter_h (int): フィルターの高さ
        filter_w (int): フィルターの幅
        stride (int, optional): ストライド
        pad (int, optional): パディング

    Returns:
        ndarray: 2次元配列
    """
    img_num, img_channel, img_height, img_width = input_data.shape
    out_height = (img_height + 2 * pad - filter_h) // stride + 1
    out_width = (img_width + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((img_num, img_channel, filter_h, filter_w, out_height, out_width))

    for y in range(filter_h):
        y_max = y + stride * out_height
        for x in range(filter_w):
            x_max = x + stride * out_width
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(img_num * out_height * out_width, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """2次元配列を入力データの形状に戻す

    Args:
        col (ndarray): 2次元配列
        input_shape (tuple): 入力データの形状((データ数, チャンネル, 高さ, 幅))
        filter_h (int): フィルターの高さ
        filter_w (int): フィルターの幅
        stride (int, optional): ストライド
        pad (int, optional): パディング

    Returns:
        ndarray: 入力データの形状に変換された2次元配列
    """
    img_num, img_channel, img_height, img_width = input_shape
    out_height = (img_height + 2 * pad - filter_h) // stride + 1
    out_width = (img_width + 2 * pad - filter_w) // stride + 1
    col = col.reshape(
        img_num, out_height, out_width, img_channel, filter_h, filter_w
    ).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros(
        (
            img_num,
            img_channel,
            img_height + 2 * pad + stride - 1,
            img_width + 2 * pad + stride - 1,
        )
    )
    for y in range(filter_h):
        y_max = y + stride * out_height
        for x in range(filter_w):
            x_max = x + stride * out_width
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : img_height + pad, pad : img_width + pad]


def preprocess(text):
    """コーパスの前処理を行う

    Args:
        text (str): テキストデータ

    Returns:
        ndarray: 単語IDの配列
        dict: 単語から単語IDへのディクショナリ
        dict: 単語IDから単語へのディクショナリ
    """
    text = text.lower()  # 小文字化
    text = text.replace(".", " .")  # .を単語として扱うため、ピリオドの前にスペースを挿入
    words = text.split(" ")  # スペースで単語を分割

    # 単語IDの付与
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # 単語IDの配列に変換
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    """コーパスから共起行列を作成する

    Args:
        corpus (ndarray): 単語IDの配列
        vocab_size (int): 語彙数
        window_size (int, optional): ウィンドウサイズ

    Returns:
        ndarray: 共起行列
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):  # corpusの単語IDを順番に取り出す
        for i in range(1, window_size + 1):  # ウィンドウサイズ分だけ左右にスライド
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:  # 左端の単語でないかチェック
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:  # 右端の単語でないかチェック
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    """コサイン類似度を計算する

    Args:
        x (ndarray): ベクトル
        y (ndarray): ベクトル
        eps (float, optional): 0除算を防ぐための微小値

    Returns:
        float: コサイン類似度
    """
    nx = x / (np.sqrt(np.sum(x**2)) + eps)  # xの正規化
    ny = y / (np.sqrt(np.sum(y**2)) + eps)  # yの正規化
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """類似した単語を検索して上位から順に表示する

    Args:
        query (str): クエリとなる単語
        word_to_id (dict): 単語から単語IDへのディクショナリ
        id_to_word (dict): 単語IDから単語へのディクショナリ
        word_matrix (ndarray): 単語ベクトルを格納した行列
        top (int, optional): 上位何件まで表示するか
    """
    # クエリの単語ベクトルを取り出す
    if query not in word_to_id:  # 単語がコーパスにないとき
        print("%s is not found" % query)
        return

    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # クエリとその他すべての単語とでコサイン類似度をそれぞれ計算する
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の結果に対して，その値が高い順に表示する
    count = 0
    for i in (-1 * similarity).argsort():  # コサイン類似度が大きい順に並び替え(iはそのインデックスを示す)
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def ppmi(co_matrix, verbose=False, eps=1e-8):
    """PPMI（正の相互情報量）を求める

    Args:
        co_matrix (ndarray): 共起行列
        verbose (bool, optional): 進捗表示するかどうか
        eps (float, optional): 0除算を防ぐための微小値

    Returns:
        ndarray: PPMI行列
    """
    ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    word_num = np.sum(co_matrix)
    appear_sum = np.sum(co_matrix, axis=0)
    total_num = co_matrix.shape[0] * co_matrix.shape[1]
    cnt = 0

    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            pmi = np.log2(
                co_matrix[i, j] * word_num / (appear_sum[j] * appear_sum[i]) + eps
            )
            ppmi_matrix[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total_num // 100 + 1) == 0:
                    print("%.1f%% done" % (100 * cnt / total_num))
    return ppmi_matrix


def create_contexts_target(corpus, window_size=1):
    """コンテキストとターゲットの作成

    Args:
        corpus (ndarray): 単語IDの配列
        window_size (int, optional): ウィンドウサイズ

    Returns:
        (ndarray, ndarray): コンテキストとターゲット
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    """one-hot表現への変換

    Args:
        corpus (ndarray): 単語IDの配列
        vocab_size (int): 語彙数

    Returns:
        ndarray: コーパスのone-hot表現
    """
    corpus_num = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((corpus_num, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    elif corpus.ndim == 2:
        context_size = corpus.shape[1]
        one_hot = np.zeros((corpus_num, context_size, vocab_size), dtype=np.int32)
        for idx0, word_ids in enumerate(corpus):
            for idx1, word_id in enumerate(word_ids):
                one_hot[idx0, idx1, word_id] = 1

    return one_hot
