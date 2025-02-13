# 文本分类实验
import os
import time

import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #实现卡号匹配
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 最大可申请显存比例
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import *
from Encoder import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pickle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.inspection import permutation_importance
from tensorflow.keras.callbacks import EarlyStopping



###################################################################################
# 使用transformer_encoder，加上了结构化信息
###################################################################################


csv_path = '../data/netbeans/netbeans.csv'
w2v_path = '../model/word2vec/netbeans/'
t2v_path = '../model/token_index/netbeans/'
model_path = '../model/trained_model/netbeans/'
pic_dir = '../pics/netbeans/'
result_txt_path = '../result/netbeans/result.txt'


def plot_and_save(epochs, train_values, val_values, title, ylabel, filename):
    plt.figure()
    plt.plot(epochs, train_values, label='Train', color='b')
    plt.plot(epochs, val_values, label='Validation', color='r')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()


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

def list_split(list, groups):
    len_list = int(len(list)) + 1
    size = int(len_list / groups) + 1
    result = []
    for i in range(0, len_list, size):
        item = list[i:i + size]
        result.append(item)

    return result


def split_data_ordered(ids, texts, labels):
    br_list = list_split(texts, 10)
    labels_list = list_split(labels, 10)
    ids_list = list_split(ids, 10)

    return br_list, labels_list, ids_list


def lists_merge(lists):
    list_merge = []
    for list in lists:
        for item in list:
            for j in item:
                list_merge.append(j)
    return list_merge

def lists_merge_test(lists):
    list_merge = []
    for list in lists:
        for item in list:
            list_merge.append(item)
    return list_merge

def split_train_test(bug_report, labels, id_list):
    br_train = lists_merge_test(bug_report[0:9])
    labels_train = lists_merge_test(labels[0:9])

    br_test = bug_report[9]
    labels_test = labels[9]
    id_test = id_list[9]

    return br_train,labels_train,br_test,labels_test,id_test

def test_model(model, br_test, label_test, id_test):
    result = model.predict(br_test)
    print("result:", result)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    predict = []
    label = []
    id = id_test
    for res, lab in zip(result, label_test):
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
    return tp, tn, fp, fn, predict, label, id


if __name__ == '__main__':

        # 1. 数据信息
        max_features = 171950
        max_len = 256
        text_len = 248
        batch_size = 128

        with open(w2v_path + 'word_index.pkl', 'rb') as pkl_word_index:
            word_index = pickle.load(pkl_word_index)
        print("len(word_index):", len(word_index))
        with open(t2v_path + 'pro_index.pkl', 'rb') as pkl_pro_index:
            pro_index = pickle.load(pkl_pro_index)
        print("len(pro_index):", len(pro_index))
        with open(t2v_path + 'com_index.pkl', 'rb') as pkl_com_index:
            com_index = pickle.load(pkl_com_index)
        print("len(com_index):", len(com_index))
        with open(t2v_path + 'ver_index.pkl', 'rb') as pkl_ver_index:
            ver_index = pickle.load(pkl_ver_index)
        with open(t2v_path + 'sev_index.pkl', 'rb') as pkl_sev_index:
            sev_index = pickle.load(pkl_sev_index)
        with open(t2v_path + 'pri_index.pkl', 'rb') as pkl_pri_index:
            pri_index = pickle.load(pkl_pri_index)
        with open(t2v_path + 'rep_index.pkl', 'rb') as pkl_rep_index:
            rep_index = pickle.load(pkl_rep_index)
        with open(t2v_path + 'ass_index.pkl', 'rb') as pkl_ass_index:
            ass_index = pickle.load(pkl_ass_index)

        print("Data downloading ... ")
        # texts, labels = load_file()
        BID, PRO, COM, VER, SEV, PRI, REP, ASS, attachment, texts, labels = load_all_file(csv_path)
        # train, test = load_dataset(word_index, texts, labels, max_len)
        print("tokenize data into index ... ")
        texts = np.array(tokenizer(texts, word_index, text_len))
        pro = np.array(stru_vec(PRO, pro_index))
        com = np.array(stru_vec(COM, com_index))
        ver = np.array(stru_vec(VER, ver_index))
        sev = np.array(stru_vec(SEV, sev_index))
        pri = np.array(stru_vec(PRI, pri_index))
        rep = np.array(stru_vec(REP, rep_index))
        ass = np.array(stru_vec(ASS, ass_index))
        att = np.array(attachment)
        print("texts shape:", texts.shape)
        print("com shape:", com.shape)
        data_concat = np.column_stack((texts, pro, com, ver, sev, pri, rep, ass, att))
        data_concat = np.array(data_concat)
        bug_report = []
        for item in data_concat:
            new_txt = []
            for data in item:
                new_txt.append(int(data))
            bug_report.append(new_txt)
        # print("bug_report:", bug_report)
        # bug_report = np.array(bug_report)
        #train_test_split
        # x_train, x_test, y_train, y_test = train_test_split(bug_report, labels, test_size=0.1)
        br_list, labels_list, ids_list = split_data_ordered(BID, bug_report, labels)
        x_train, y_train, x_test, y_test,id_test = split_train_test(br_list, labels_list, ids_list)
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)


        # 检查是否有保存的模型
        if os.path.exists(model_path):
            print("加载已有模型...")
            model = load_model(model_path)
        else:
            print("训练新模型...")
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
                out_seq = Encoder(12, 128, 4, 256, max_len)(embeddings, mask_inputs)

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
                         epochs=5,
                         validation_split=0.1)
                # 保存模型
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)

        tp, tn, fp, fn, predict, label, id = test_model(model, x_test, y_test, id_test)

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
        result = pd.DataFrame([{'bug_id': i, 'predict': p, 'label': l} for i, p, l in zip(id, predict, label)],
                              columns=['bug_id', 'predict', 'label'])
        # result.to_csv(result_csv_path, index=False)
        file = open(result_txt_path, 'a+')
        file.write('\n' + "The results:" + '\n' + "auc:" + str(roc_auc) + '\n' + "tp:" + str(tp) + '\n' + "fn:" + str(fn) + '\n'
                   + "fp:" + str(fp) + '\n' + "tn:" + str(tn) + '\n' + "accuracy:" + str(accuracy) + '\n'
                   + "precision:" + str(precision) + '\n' + "recall:" + str(recall) + '\n' + "f1_score:" + str(f1_score) + '\n')
        file.close()
