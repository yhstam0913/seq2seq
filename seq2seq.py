# coding: utf-8
"""
Sequence to Sequenceのchainer実装
Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le.,
"Sequence to sequence learning with neural networks.",
Advances in neural information processing systems. 2014.

詳細
http://qiita.com/kenchin110100/items/b34f5106d5a211f4c004
"""

import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import datetime
import random
import json
import sys
import MeCab
from collections import defaultdict

class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Encoder, self).__init__(
            # 単語を単語ベクトルに変換する層
            xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            eh = links.Linear(embed_size, 4 * hidden_size),
            # 出力された中間層を4倍のサイズに変換するための層
            hh = links.Linear(hidden_size, 4 * hidden_size)
        )
    

    def __call__(self, x, c, h):
        """
        :param x: ミニバッチ中のある順番の単語IDリスト
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        # xeで単語ベクトルに変換して、そのベクトルをtanhにかけ，-1~1に圧縮処理        
        # x:BATCH_SIZEのベクトル
        # self.xe(x):BATCH_SIZE*EMBED_SIZEの行列
        # EMBED次元の単語ベクトルがBATCHSIZE個並んでいる
        # xの各要素をEMBEDのベクトルで表現し，そのベクトルをならべたもの
        # → 各発話の単語ベクトルを並べたもの
        e = functions.tanh(self.xe(x))
        # 前の内部メモリの値と単語ベクトルの4倍サイズ、中間層の4倍サイズを足し合わせて入力
        return functions.lstm(c, self.eh(e) + self.hh(h))

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Decoder, self).__init__(
            # 入力された単語を単語ベクトルに変換する層
            ye = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            eh = links.Linear(embed_size, 4 * hidden_size),
            # 中間ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            hh = links.Linear(hidden_size, 4 * hidden_size),
            # 出力されたベクトル(中間ベクトル)を単語ベクトルのサイズに変換する層
            he = links.Linear(hidden_size, embed_size),
            # 単語ベクトルを語彙サイズのベクトル（one-hotなベクトル）に変換する層
            ey = links.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        """
        :param y: one-hotな単語
        :param c: 内部メモリ
        :param h: 隠れそう
        :return: 予測単語、次の内部メモリ、次の隠れ層
        """
        # yeで単語ベクトルに変換して、そのベクトルをtanhにかけ，-1~1に圧縮処理        
        # y:BATCH_SIZEのベクトル
        # self.ye(y):BATCH_SIZE*EMBED_SIZEの行列
        # EMBED次元の単語ベクトルがBATCHSIZE個並んでいる
        # yの各要素をEMBEDのベクトルで表現し，そのベクトルをならべたもの
        # → 各応答の単語ベクトルを並べたもの
        e = functions.tanh(self.ye(y))
        c, h = functions.lstm(c, self.eh(e) + self.hh(h))
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h

class Seq2Seq(Chain):
    def __init__(self, vocab_size, enc_embed_size, dec_embed_size, hidden_size, batch_size):
        """
        Seq2Seqの初期化
        :param vocab_size: 語彙サイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 中間ベクトルのサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Seq2Seq, self).__init__(
            # Encoderのインスタンス化
            encoder = LSTM_Encoder(vocab_size, enc_embed_size, hidden_size),
            # Decoderのインスタンス化
            decoder = LSTM_Decoder(vocab_size, dec_embed_size, hidden_size)
        )
        self.vocab_size     = vocab_size
        self.enc_embed_size = enc_embed_size
        self.dec_embed_size = dec_embed_size
        self.hidden_size    = hidden_size
        self.batch_size     = batch_size
        # GPUで計算する場合はcupyをCPUで計算する場合はnumpyを使う
        self.ARR = cuda.cupy if FLAG_GPU else np

    def encode(self, words):
        """
        Encoderを計算する部分
        :param words: 単語が記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        # エンコーダーに単語を順番に読み込ませる
        # エンコーダーの返り値c,hを次のエンコーダーへの入力として読み込ませる
        for w in words:
            c, h = self.encoder(w, c, h)

        # 計算した中間ベクトルをデコーダーに引き継ぐためにインスタンス変数にする
        self.h = h
        # 内部メモリはデコーダーに引き継がないので、初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        デコーダーを計算する部分
        :param w: 単語
        :return: 単語数サイズのベクトルを出力する
        """
        # 内部メモリ，隠れ層は保存
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t

    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        :return:
        """
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        self.zerograds()

def forward(enc_words, dec_words, model, ARR):
    """
    順伝播の計算を行う関数
    :param enc_words: 発話文の単語を記録したリスト
    :param dec_words: 応答文の単語を記録したリスト
    :param model: Seq2Seqのインスタンス
    :param ARR: cuda.cupyかnumpyか
    :return: 計算した損失の合計
    """
    # バッチサイズを記録
    # 転置行列に成っているのでlen(enc_words[0])がバッチサイズと等しい
    batch_size = len(enc_words[0])
    # print("\t",len(enc_words),batch_size)
    # model内に保存されている勾配をリセット
    model.reset()

    # 発話リスト内の単語を、chainerの型であるVariable型に変更
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    # エンコードの計算 ①
    model.encode(enc_words)
    # 損失の初期化
    loss = Variable(ARR.zeros((), dtype='float32'))
    # <eos>をデコーダーに読み込ませる ②
    t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
    # デコーダーの計算
    for w in dec_words:
        # 1単語ずつをデコードする ③
        # 最初のループは<eos>を読み込ませる
        # 以降は正解単語を読み込ませる
        y = model.decode(t)
        # 正解単語wをVariable型正解単語tに変換
        t = Variable(ARR.array(w, dtype='int32'))
        # 正解単語tと予測単語yを照らし合わせて損失を計算 ④
        loss += functions.softmax_cross_entropy(y, t)
    return loss

# forward_testを書き換えたsample
# 入力を学習時と同じ次元(batchsize)に変更
def forward_test_sample(enc_words, model, ARR):
    ret = []
    model.reset()
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
    model.encode(enc_words)
    t = Variable(ARR.array([0 for _ in range(BATCH_SIZE)], dtype='int32'))
    counter = 0
    while counter < 50:
        y = model.decode(t)
        label = y.data.argmax()
        ret.append(label)
        t = Variable(ARR.array([label for _ in range(BATCH_SIZE)], dtype='int32'))
        counter += 1
        # ID0がきたら終わり
        if label == 0:
            counter = 50
    return ret

# trainの関数
def make_minibatch(minibatch):
    # 転置するとインデックス0からとりだせるようになる
    # 転置後enc_words
    # [?  　,...]
    # [好き　,...]
    # [が   ,...]
    # [何　　,...]
    # 転置後dec_words
    # [肉　 ,...]
    # [が　 ,...]
    # [好き ,...]
    # [です ,...]
    # [EOS ,...]

    # enc_wordsの作成
    enc_words = [row[0] for row in minibatch]
    enc_max = np.max([len(row) for row in enc_words])
    # ベクトルを固定長にするためにenc_maxの長さにあわせて足りない部分を-1で埋める
    # Encoderに順に読み込ませるときに逆順に読み込ませる方がいいことが知られている
    # そのため-1で埋めるときは左から-1を埋めていく．    
    enc_words = np.array([[-1]*(enc_max - len(row)) + row for row in enc_words], dtype='int32')
    enc_words = enc_words.T

    # dec_wordsの作成
    dec_words = [row[1] for row in minibatch]
    dec_max = np.max([len(row) for row in dec_words])
    # ベクトルを固定長にするために-1で埋める
    # EncoderにたいしてDecoderでは右から足りない部分を-1で埋めていく
    dec_words = np.array([row + [-1]*(dec_max - len(row)) for row in dec_words], dtype='int32')
    dec_words = dec_words.T

    return enc_words, dec_words

def test():
    
    # 語彙数
    vocab_size = len(word_to_id)

    model = Seq2Seq(vocab_size=vocab_size,
                    enc_embed_size=ENC_EMBED_SIZE,
                    dec_embed_size=DEC_EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=BATCH_SIZE)

    ARR = cuda.cupy if FLAG_GPU else np
    # モデルの読み込み
    serializers.load_hdf5(INPUT_MODEL + '.weights',model)
    
    m = MeCab.Tagger("-Owakati")
    while True:
        print("Try talking !!")
        # 入力文を分かち書き
        input_sentence = input()
        input_wakati_list = m.parse(input_sentence).split()
        # BATCH_SIZEと同じ次元の単語IDベクトルを作成
        input_id_list = []
        for word in input_wakati_list:
            if word in word_to_id:
                temp = [word_to_id[word]for _ in range(BATCH_SIZE) ]
                input_id_list.append(temp)
            else:
                print("{}という単語はコーパス中にはありません".format(word))
                break
        # 順伝播させて応答IDリストを獲得
        else:
            ret = forward_test_sample(input_id_list,model,ARR)
            # IDを単語に対応させる
            for word_id in ret:
                print(id_to_word[word_id],end = " ") 
            print()

def train():

    # 語彙数
    vocab_size = len(word_to_id)
    # モデルのインスタンス化
    model = Seq2Seq(vocab_size     = vocab_size,
                    enc_embed_size = ENC_EMBED_SIZE,
                    dec_embed_size = DEC_EMBED_SIZE,
                    hidden_size    = HIDDEN_SIZE,
                    batch_size     = BATCH_SIZE)
    # モデルの初期化
    model.reset()
    # GPUのセット
    if FLAG_GPU:
        ARR = cuda.cupy
        cuda.get_device(0).use()
        model.to_gpu(0)
    else:
        ARR = np

    # 学習開始
    for epoch in range(EPOCH_NUM):
        # エポックごとにoptimizerの初期化
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))

        random.shuffle(data)
        for num in range(len(data)//BATCH_SIZE):    
            # 任意のサイズのミニバッチを作成
            minibatch = data[num*BATCH_SIZE: (num+1)*BATCH_SIZE]
            # 読み込み用のデータ作成
            enc_words, dec_words = make_minibatch(minibatch)
            # modelのリセット
            model.reset()
            # 順伝播で損失を計算
            total_loss = forward(enc_words=enc_words,
                                 dec_words=dec_words,
                                 model=model,
                                 ARR=ARR)
            # 誤差逆伝播で勾配の計算
            total_loss.backward()
            # 計算したネットワークを使ってネットワークを更新
            opt.update()
            # 記録された勾配を初期化
            opt.zero_grads()
            
        print ('Epoch %s 終了' % (epoch+1))
        # モデルの保存
        outputpath = OUTPUT_PATH%(ENC_EMBED_SIZE, DEC_EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, epoch+1)
        serializers.save_hdf5(outputpath, model)

def pre_processing(utt_file,res_file):

    data = list()
    utt_id_list = list()
    res_id_list = list()
    word_to_id = defaultdict(int)

    utt_lines = open(utt_file).read().split("\n")
    res_lines = open(res_file).read().split("\n")

    m = MeCab.Tagger("-Owakati")
    for utt,res in zip(utt_lines,res_lines):

        utt_wakati_list = m.parse(utt).split()
        res_wakati_list = m.parse(res).split()
        
        # 発話をID化
        for word in utt_wakati_list:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id) + 2
            utt_id_list.append(word_to_id[word])

        # 応答をID化
        for word in res_wakati_list:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id) + 2
            res_id_list.append(word_to_id[word])

        # EOS
        res_id_list.append(0)
        data.append([utt_id_list,res_id_list])
        utt_id_list = []
        res_id_list = []

    # idに対応する単語が格納された辞書を用意
    id_to_word = {v:k for k, v in word_to_id.items()}
    id_to_word[0],id_to_word[-1] = '<EOS>',''

    return data,word_to_id,id_to_word

if __name__ == "__main__":
    '''
    train: python3 seq2seq.py train
    test : python3 seq2seq.py test s2s_model_file
    '''
    # 発話が1行ごとに格納されたテキストファイル
    UTT_FILE_PATH  = 'utterance.txt'
    # 発話に対する応答が格納されたテキストファイル
    RES_FILE_PATH  = 'response.txt'
    # モデルの出力ファイル名
    OUTPUT_PATH    = 's2s_models/ENCEMBED%s_DECEMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.weights'
    # encoderに単語を入力するときの単語ベクトルの次元数
    ENC_EMBED_SIZE = 300
    # decoderから単語を出力するときの単語ベクトルの次元数
    DEC_EMBED_SIZE = 300 
    # 隠れ層の次元数
    HIDDEN_SIZE    = 300
    # ミニバッチのサイズ
    BATCH_SIZE     = 2
    # 学習回数
    EPOCH_NUM      = 30

    print ('start pre processing: ', datetime.datetime.now())
    data, word_to_id, id_to_word = pre_processing(UTT_FILE_PATH,RES_FILE_PATH)
    print ('end pre processing: ', datetime.datetime.now())

    if sys.argv[1] == "train":
        FLAG_GPU = True
        print ('start training: ', datetime.datetime.now())
        train()
        print ('end training: ', datetime.datetime.now())

    elif sys.argv[1] == "test":
        FLAG_GPU = False
        INPUT_MODEL = sys.argv[2]
        test()
    else:
        print("train or test only")