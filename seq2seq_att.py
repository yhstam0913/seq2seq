# coding: utf-8
"""
Seq2Seq + Global-Attention
Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
"Neural machine translation by jointly learning to align and translate."
arXiv preprint arXiv:1409.0473 (2014).

詳細
http://qiita.com/kenchin110100/items/eb70d69d1d65fb451b67
"""

import numpy as np
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import datetime
import random
import json
import sys
import MeCab
from collections import defaultdict
import matplotlib.pyplot as plt

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

class Att_LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        Attention ModelのためのDecoderのインスタンス化
        :param vocab_size: 語彙数
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(Att_LSTM_Decoder, self).__init__(
            # 単語を単語ベクトルに変換する層
            ye=links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            eh=links.Linear(embed_size, 4 * hidden_size),
            # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            hh=links.Linear(hidden_size, 4 * hidden_size),
            # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            fh=links.Linear(hidden_size, 4 * hidden_size),
            # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
            bh=links.Linear(hidden_size, 4 * hidden_size),
            # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
            he=links.Linear(hidden_size, embed_size),
            # 単語ベクトルを語彙数サイズのベクトルに変換する層
            ey=links.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h, f, b):
        """
        Decoderの計算
        :param y: Decoderに入力する単語
        :param c: 内部メモリ
        :param h: Decoderの中間ベクトル
        :param f: Attention Modelで計算された順向きEncoderの加重平均
        :param b: Attention Modelで計算された逆向きEncoderの加重平均
        :return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
        """
        # 単語を単語ベクトルに変換
        e = functions.tanh(self.ye(y))
        # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTMを更新
        c, h = functions.lstm(c, self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b))
        # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h

class Attention(Chain):
    def __init__(self, hidden_size):
        """
        Attentionのインスタンス化
        :param hidden_size: 隠れ層のサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Attention, self).__init__(
            # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            fh=links.Linear(hidden_size, hidden_size),
            # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            bh=links.Linear(hidden_size, hidden_size),
            # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
            hh=links.Linear(hidden_size, hidden_size),
            # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
            hw=links.Linear(hidden_size, 1),
        )
        # 隠れ層のサイズを記憶
        self.hidden_size = hidden_size
        # GPUを使う場合はcupyを使わないときはnumpyを使う
        self.ARR = cuda.cupy if FLAG_GPU else np
        self.weights = []

    def __call__(self, fs, bs, h):
        """
        Attentionの計算
        :param fs: 順向きのEncoderの隠れ層が記録されたリスト
        :param bs: 逆向きのEncoderの隠れ層が記録されたリスト
        :param h: Decoderで出力された中間ベクトル
        :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
        """
        # ミニバッチのサイズを記憶
        batch_size = h.data.shape[0]
        # 荷重aを記録するためのリストの初期化
        a_list = []
        # ウェイトの合計値を計算するための値を初期化
        sum_a = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))

        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = functions.tanh(self.fh(f)+self.bh(b)+self.hh(h))
            # softmax関数を使って正規化する
            # a:バッチサイズと同じ次元数
            # a = exp(eij) ある時刻Jの入力i番目の隠れ層
            #         発話1
            # a = [ [0.945...] [1.0054...] [...] [...] [...] ... ]
            a = functions.exp(self.hw(w))
            # 計算したウェイトを記録
            a_list.append(a)
            # sum_a = Σexp(eik)
            sum_a += a

        # 出力する加重平均ベクトルの初期化
        att_f = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        att_b = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        
        word_weights = []
        for f, b, a in zip(fs, bs, a_list):
            # ウェイトの和が1になるように正規化
            # a = αij
            a /= sum_a
            # 時刻tの単語を予測する際の荷重を保存
            word_weights.append(float(a[0].data))
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            # batch_matmul:行列の掛け算
            # ci = Σαij*hj
            att_f += functions.reshape(functions.batch_matmul(f, a), (batch_size, self.hidden_size))
            att_b += functions.reshape(functions.batch_matmul(b, a), (batch_size, self.hidden_size))
        
        # 書く単語を予測する際の荷重を保存
        self.weights.append(word_weights)
        # print(word_weights)
        return att_f, att_b, self.weights

class Att_Seq2Seq(Chain):
    def __init__(self, vocab_size, enc_embed_size, dec_embed_size, hidden_size, batch_size):
        """
        Seq2Seq + Attentionのインスタンス化
        :param vocab_size: 語彙数のサイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Att_Seq2Seq, self).__init__(
            # 順向きのEncoder
            f_encoder = LSTM_Encoder(vocab_size, enc_embed_size, hidden_size),
            # 逆向きのEncoder
            b_encoder = LSTM_Encoder(vocab_size, enc_embed_size, hidden_size),
            # Attention Model
            attention = Attention(hidden_size),
            # Decoder
            decoder = Att_LSTM_Decoder(vocab_size, dec_embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.enc_embed_size = enc_embed_size
        self.dec_embed_size = dec_embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # GPUで計算する場合はcupyをCPUで計算する場合はnumpyを使う
        self.ARR = cuda.cupy if FLAG_GPU else np

        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fs = []
        self.bs = []

    def encode(self, words):
        """
        Encoderの計算
        :param words: 入力で使用する単語記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # 順向きでEncoderの計算
        for w in words:
            c, h = self.f_encoder(w, c, h)
            # 計算された中間ベクトル(隠れ層)を記録
            self.fs.append(h)

        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        
        # 逆向きでEncoderの計算
        # やってることはreverseしてencoderに渡すだけ
        for w in reversed(words):
            c, h = self.b_encoder(w, c, h)
            # 計算された中間ベクトル(隠れ層)を記録
            # 隠れ層を保管し先頭に追加していく
            self.bs.insert(0, h)

        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        Decoderの計算
        :param w: Decoderで入力する単語
        :return: 予測単語
        """
        # 隠れ層が格納されたリストと直前の隠れ層を渡す
        # アテンションは前の単語の情報は使わない
        att_f, att_b, weights = self.attention(self.fs, self.bs, self.h)
        # 隠れ層を順に更新していく
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        return t, weights

    def reset(self):
        """
        インスタンス変数を初期化する
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        # Encoderの中間ベクトルを記録するリストの初期化
        self.fs = []
        self.bs = []
        self.attention.weights = []
        # 勾配の初期化
        self.zerograds()

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
        y,weights = model.decode(t)
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
        y,weights = model.decode(t)
        label = y.data.argmax()
        ret.append(label)
        t = Variable(ARR.array([label for _ in range(BATCH_SIZE)], dtype='int32'))
        counter += 1
        # ID0(EOS)がきたら終わり
        if label == 0:
            counter = 50

    return ret,weights

def ja_test():
    
    # 語彙数
    vocab_size = len(word_to_id)

    model = Att_Seq2Seq(vocab_size=vocab_size,
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
            ret,weights = forward_test_sample(input_id_list,model,ARR)
            
            # IDを単語に対応させる
            for word_id in ret:
                print(id_to_word[word_id],end = " ") 
            print()
            save_heatmap(input_wakati_list,ret,weights)

def en_test():
    
    # 語彙数
    vocab_size = len(word_to_id)

    model = Att_Seq2Seq(vocab_size=vocab_size,
                    enc_embed_size=ENC_EMBED_SIZE,
                    dec_embed_size=DEC_EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=BATCH_SIZE)

    ARR = cuda.cupy if FLAG_GPU else np
    # モデルの読み込み
    serializers.load_hdf5(INPUT_MODEL + '.weights',model)
    
    while True:
        print("Try talking !!")
        # 入力文を分かち書き
        input_sentence = input()
        input_wakati_list = input_sentence.split(" ")
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
            ret,weights = forward_test_sample(input_id_list,model,ARR)
            
            # IDを単語に対応させる
            for word_id in ret:
                print(id_to_word[word_id],end = " ") 
            print()
            save_heatmap(input_wakati_list,ret,weights)

def train():

    # 語彙数
    vocab_size = len(word_to_id)
    # モデルのインスタンス化
    model = Att_Seq2Seq(
                    vocab_size     = vocab_size,
                    enc_embed_size = ENC_EMBED_SIZE,
                    dec_embed_size = DEC_EMBED_SIZE,
                    hidden_size    = HIDDEN_SIZE,
                    batch_size     = BATCH_SIZE
                    )
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
            total_loss = forward(enc_words = enc_words,
                                 dec_words = dec_words,
                                 model     = model,
                                 ARR       = ARR
                                 )
            # 誤差逆伝播で勾配の計算
            total_loss.backward()
            # 計算したネットワークを使ってネットワークを更新
            opt.update()
            # 記録された勾配を初期化
            opt.zero_grads()
            print('{}/{}終了'.format(num,len(data)//BATCH_SIZE))
        print ('Epoch %s 終了' % (epoch+1))
        # モデルの保存
        outputpath = OUTPUT_PATH%(ENC_EMBED_SIZE, DEC_EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, epoch+1)
        serializers.save_hdf5(outputpath, model)

def save_heatmap(input_wakati_list,ret,weights):

    # weights = np.array(list(reversed(weights)))
    weights = np.array(weights).T[::-1]
    ret_labels = [ id_to_word[_id] for _id in ret ]
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(weights,cmap=plt.cm.Reds)
    fig.colorbar(heatmap,ax = ax)

    # print(weights)
    # print(np.arange(weights.shape[1]))

    ax.set_yticks(np.arange(weights.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(weights.shape[1]) + 0.5, minor=False)

    # ax.set_yticklabels(list(reversed(input_wakati_list)), minor=False)
    ax.set_yticklabels(list(reversed(input_wakati_list)), minor=False)
    ax.set_xticklabels(ret_labels, minor=False)

    plt.savefig('heatmap.jpg')
    
    pass

def ja_pre_processing(utt_file,res_file):

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

def en_pre_processing(utt_file,res_file):

    data = list()
    utt_id_list = list()
    res_id_list = list()
    word_to_id = defaultdict(int)

    utt_lines = open(utt_file).read().split("\n")
    res_lines = open(res_file).read().split("\n")

    for utt,res in zip(utt_lines,res_lines):

        utt_wakati_list = utt.split(" ")
        res_wakati_list = res.split(" ")

        if len(utt_wakati_list) > MAX_LENGTH or len(res_wakati_list) > MAX_LENGTH:
            continue
        
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
    train: python3 seq2seq_w2v.py train
    test : python3 seq2seq_w2v.py test s2s_model_file
    '''
    # 発話が1行ごとに格納されたテキストファイル
    # UTT_FILE_PATH  = 'utterance.txt'
    UTT_FILE_PATH  = 'utt_cmdc_100K.txt'
    # 発話に対する応答が格納されたテキストファイル
    # RES_FILE_PATH  = 'response.txt'
    RES_FILE_PATH  = 'res_cmdc_100K.txt'
    # モデルの出力ファイル名
    OUTPUT_PATH    = 's2s_att_cmdc_models/ENCEMBED%s_DECEMBED%s_HIDDEN%s_BATCH%s_EPOCH%s.weights'
    # encoderに単語を入力するときの単語ベクトルの次元数
    ENC_EMBED_SIZE = 100
    # decoderから単語を出力するときの単語ベクトルの次元数
    DEC_EMBED_SIZE = 100
    # 隠れ層の次元数
    HIDDEN_SIZE = 300
    # ミニバッチのサイズ
    BATCH_SIZE  = 100
    # 学習回数
    EPOCH_NUM   = 100
    # 学習データの最大長
    MAX_LENGTH  =  15

    print ('start pre processing: ', datetime.datetime.now())
    data, word_to_id, id_to_word = en_pre_processing(UTT_FILE_PATH,RES_FILE_PATH)
    # data, word_to_id, id_to_word = ja_pre_processing(UTT_FILE_PATH,RES_FILE_PATH)
    print ('end pre processing  : ', datetime.datetime.now())

    if sys.argv[1] == "train":
        FLAG_GPU = True
        print ('start training: ', datetime.datetime.now())
        print ('EPOCH_NUM  : {}'.format(EPOCH_NUM))
        print ('ENC_EMBED  : {}'.format(ENC_EMBED_SIZE))
        print ('ENC_EMBED  : {}'.format(ENC_EMBED_SIZE))
        print ('HIDDEN_SIZE: {}'.format(HIDDEN_SIZE))
        print ('BATCHS_IZE : {}'.format(BATCH_SIZE))
        print ('DATA_SIZE  : {}'.format(len(data)))
        train()
        print ('end training: ', datetime.datetime.now())

    elif sys.argv[1] == "test":
        FLAG_GPU = False
        INPUT_MODEL = sys.argv[2]
        # ja_test()
        en_test()
    else:
        print("train or test only")