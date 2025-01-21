# 文本分类实验
import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 最大可申请显存比例
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from Encoder import *
import pandas as pd

from sklearn.model_selection import train_test_split

import pickle
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping


csv_path = '../data/eclipse_rm_dup/data1_3.71.csv'
w2v_path = '../valid-bug-report/model/word2vec/eclipse/'
t2v_path = '../data/valid-bug-report/model/token_index/'


def load_all_file(csv_path):
    dataFrame = pd.read_csv(csv_path, encoding='utf-8')
    # print("dataFrame:", dataFrame)
    BID = []
    PRO = []
    COM = []
    VER = []
    SEV = []
    PRI = []
    REP = []
    ASS = []
    attachment = []
    texts = []
    labels = []  # 存储读取的y
    # 遍历 获取数据
    for i in range(len(dataFrame)):
        BID.append(dataFrame.at[i,'BID'])
        PRO.append(dataFrame.at[i, 'PRO'])
        COM.append(dataFrame.at[i,'COM'])
        VER.append(dataFrame.at[i, 'VER'])
        SEV.append(dataFrame.at[i, 'SEV'])
        PRI.append(dataFrame.at[i, 'PRI'])
        REP.append(dataFrame.at[i, 'REP'])
        ASS.append(dataFrame.at[i, 'ASS'])
        attachment.append(dataFrame.at[i, 'attachment'])

        summary = str(dataFrame.at[i, 'summary'])
        description = str(dataFrame.at[i, 'description'])
        text = summary + description
        texts.append(text)
        labels.append(dataFrame.at[i, 'valid'])  # 每个元素为一个int 代表类别 # [2, 6, ... 3] 的形式。减一为了 从0开始

    return BID, PRO, COM, VER, SEV, PRI, REP, ASS, attachment, texts, labels

def load_file():
        dataFrame = pd.read_csv(csv_path, encoding='utf-8')
        texts= []
        labels = []  # 存储读取的y
        # ids = []
        # 遍历 获取数据
        for i in range(len(dataFrame)):
            # ids.append(dataFrame.at[i,'bug_id'])

            summary = str(dataFrame.at[i, 'summary'])
            description = str(dataFrame.at[i, 'description'])
            text = summary + description
            texts.append(text)
            labels.append(dataFrame.at[i, 'valid'])  # 每个元素为一个int 代表类别 # [2, 6, ... 3] 的形式。减一为了 从0开始

        return texts, labels  # 总文本，总标签

def tokenizer(texts, word_index, MAX_SEQUENCE_LENGTH):
        data = []
        for sentence in texts:
            if sentence is np.nan:
                sentence = ' '
            new_txt = []
            for word in sentence.split(' '):
                try:
                    new_txt.append(word_index[word])  # 把句子中的 词语转化为index
                except:
                    new_txt.append(0)
            data.append(new_txt)
        # print("data:",data)
        # time.sleep(20)
        texts = sequence.pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
        return texts

def stru_vec(strudata,token_index):
    result = []
    for data in strudata:
        if data is np.nan:
            data = ' '
        new_txt = []
        try:
            new_txt.append(token_index[data])  # 把句子中的 词语转化为index
        except:
            new_txt.append(0)
        result.append(new_txt)
    # print("result:",result)
    # result = sequence.pad_sequences(result, maxlen=MAX_SEQUENCE_LENGTH)  # 使用kears的内置函数padding对齐句子,好处是输出numpy数组，不用自己转化了
    return result


def load_dataset(word_index, texts, labels, max_len):
    texts = np.array(tokenizer(texts, word_index, max_len))

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)


    return (x_train,  y_train), (x_test,  y_test)


if __name__ == '__main__':

        # 1. 数据信息
        max_features = 71950
        max_len = 256
        batch_size = 16

        with open(w2v_path + 'word_index.pkl', 'rb') as pkl_word_index:
            word_index = pickle.load(pkl_word_index)
        print("len(word_index):", len(word_index))

        print("Data downloading ... ")
        texts, labels = load_file()
        train, test = load_dataset(word_index, texts, labels, max_len)
        x_train, y_train = train
        x_test, y_test = test
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)


        # 2. 构造模型，及训练模型
        with tf.device('/GPU:0'):
            inputs = Input(shape=(256,), dtype='int32')
            embeddings = Embedding(max_features, 128)(inputs)

            print("\n"*2)
            print("embeddings:")
            print(embeddings)

            mask_inputs = padding_mask(inputs)
            print("mask_inputs:")
            print(mask_inputs)
            out_seq = Encoder(2, 128, 4, 256, max_len)(embeddings, mask_inputs)

            print("\n"*2)
            print("out_seq:")
            print(out_seq)

            out_seq = GlobalAveragePooling1D()(out_seq)

            print("\n"*2)
            print("out_seq:")
            print(out_seq)

            out_seq = Dropout(0.3)(out_seq)
            out_seq = Dropout(0.3)(out_seq)
            out_seq = Dropout(0.3)(out_seq)

            outputs = Dense(1, activation='sigmoid', name='fullyConn')(out_seq)

            model = Model(inputs=inputs, outputs=outputs)
            print(model.summary())

            opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            loss = 'binary_crossentropy'
            model.compile(loss=loss,
                         optimizer=opt,
                         metrics=['accuracy'])

            es = EarlyStopping(patience=5)

            print('Train...')
            history = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=3,
                     validation_split=0.1)
            result = model.predict(x_test)
            print("result:", result)
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            predict = []
            label = []
            # id = id_test
            for res, lab in zip(result, y_test):
                predict.append(res[0])
                label.append(int(lab))
                if res[0] >= 0.50 and int(lab) == 1:
                    tp = tp + 1
                if res[0] >= 0.50 and int(lab) == 0:
                    fp = fp + 1
                if res[0] < 0.50 and int(lab) == 1:
                    fn = fn + 1
                if res[0] < 0.50 and int(lab) == 0:
                    tn = tn + 1

            fpr, tpr, threshold = roc_curve(label, predict)  ###计算真正率和假正率
            roc_auc = auc(fpr, tpr)
            print('-----------------------------')
            print('The results:')
            print('auc: ' + str(roc_auc))
            print('tp: ' + str(tp))
            print('fn: ' + str(fn))
            print('fp: ' + str(fp))
            print('tn: ' + str(tn))
            accuracy = (tp+tn)/(tp+fn+tn+fp)
            print('accuracy: ' + str(accuracy))
            precision = tp / (tp + fp)
            print('precision: ' + str(precision))
            recall = tp / (tp + fn)
            print('recall: ' + str(recall))
            f1_score = 2 * precision * recall / (precision + recall)
            print('f1_score: ' + str(f1_score))