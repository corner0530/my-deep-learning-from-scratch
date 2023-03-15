from nltk.corpus import wordnet

if __name__ == "__main__":
    # どれだけ異なる意味が存在するのかを確認
    # リストの要素数だけ，異なる同義語グループがある
    # 各要素は`(単語名).(属性).(グループのインデックス)`
    print(wordnet.synsets("car"))

    # 同義語グループの持つ意味を確認
    car = wordnet.synset("car.n.01")  # 同義語グループ
    print(car.definition())

    # 同義語グループにどのような単語が存在するのかを見る
    print(car.lemma_names())

    # 他の単語との意味的な上位・下位の関係を見る
    # 上位語からその単語までの経路(一般に複数ある)を見る
    print(car.hypernym_paths()[0])

    # 単語間の類似度を算出する
    car = wordnet.synset("car.n.01")
    novel = wordnet.synset("novel.n.01")
    dog = wordnet.synset("dog.n.01")
    motorcycle = wordnet.synset("motorcycle.n.01")

    print(car.path_similarity(novel))
    print(car.path_similarity(dog))
    print(car.path_similarity(motorcycle))
