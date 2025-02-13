from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import itertools
import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Lambda, concatenate, Flatten, Dropout, Dense, Activation, Bidirectional, CuDNNLSTM, BatchNormalization
import keras.backend as K
from keras.optimizers import Adadelta, adam

from keras.models import load_model
import tensorflow as tf
import json
from modules import text_to_word_list
import sys, os, pickle

from imblearn.combine import SMOTETomek

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
sess = tf.Session(config = config)

# Project data
PROJECT = "eclipse"
MODEL_NO = 1
DATA_CSV = 'data/' + PROJECT + '.csv'

# Word embedding vector
WORD_EMBEDDING_DIM = '100'
EMBEDDING_ALGO = 'sg'
WORD_EMBEDDING_FILE = 'data/pretrained_embeddings/word2vec/'+ PROJECT + '_bugreport-vectors-gensim-' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM +'dwin10.bin'

# HIN embedding vector
HIN_EMBEDDING_DIM = '128'
HIN_EMBEDDING_FILE = 'data/pretrained_embeddings/hin2vec/' + PROJECT + '_node_' + HIN_EMBEDDING_DIM + 'd_5n_4w_1280l.vec'
HIN_NODE_DICT = 'data/hin_node_dict/' + PROJECT + '_node.dict'

# Model Save
MODEL_SAVE_FILE = 'output/trained_model/' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA

# Model Training history record
EXP_HISTORY_ACC_SAVE_FILE = 'output/training_history/' + 'acc_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
EXP_HISTORY_VAL_ACC_SAVE_FILE = 'output/training_history/' + 'val_acc_'+ PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
EXP_HISTORY_LOSS_SAVE_FILE = 'output/training_history/' + 'loss_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
EXP_HISTORY_VAL_LOSS_SAVE_FILE = 'output/training_history/' + 'val_loss_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 

# Model Test history record
EXP_TEST_HISTORY_FILE = 'output/training_history/' + 'test_result_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 

# Result of bug report prediction
PREDICTION_RESULT_FILE = 'output/training_result/' + PROJECT + '_predict_result'
# Read data
# Load training and test set
data_df = pd.read_csv(DATA_CSV)
# Initialize hin features
data_df['hin'] = 'nan'


# Preprocess the text information
stops = set(stopwords.words('english'))

##### Prepare word embedding -- Summary
vocabulary = dict()
inverse_vocabulary = ['<unk>']
word2vec = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_FILE, binary=True)

text_cols = ['summary', 'description']

# Iterate over the summaries
for index, row in data_df.iterrows():
    # Iterate through the text of both summaries of the row
    for text in text_cols:
        s2n = []
        for word in text_to_word_list(row[text]):
            # Check for unwanted words
            if word in stops and word not in word2vec.vocab:
                continue

            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                s2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                s2n.append(vocabulary[word])

        # Replace summaries as word to summary as number representaion
        data_df.at[index, text] = s2n

# Word embedding matrix settings
word_embedding_dim = int(WORD_EMBEDDING_DIM)
word_embeddings = 1 * np.random.randn(len(vocabulary) + 1, word_embedding_dim)
word_embeddings[0] = 0

# Build the word embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        word_embeddings[index] = word2vec.word_vec(word)


del word2vec


# Prepare hin embedding
hin_vocabulary = set()
hin_cols1 = ["bid","component","product","reporter"]

with open(HIN_NODE_DICT, 'r') as f:
    hin_node_dict = json.load(f)

for index, row in data_df.iterrows():
    s2n = []
    for hin in hin_cols1:
        if str(row[hin]) != 'nan':
            hin_node_id = hin_node_dict[str(row[hin])][0]
            hin_vocabulary.add(hin_node_id)
            s2n.append(hin_node_id) 
        else:
            s2n.append(0)
    data_df.at[index,'hin'] = s2n

# HIN embedding matrix settings
hin_embedding_dim = int(HIN_EMBEDDING_DIM)
hin_embeddings = 1 * np.random.randn(max(hin_vocabulary) + 1, hin_embedding_dim)
hin_embeddings[0] = 0

# Load hin node2vec
node2vec = {}
with open(HIN_EMBEDDING_FILE) as f:
    first = True
    for line in f:
        if first:
            first = False
            continue
        line = line.strip()
        tokens = line.split(' ')
        node2vec[tokens[0]] = np.array(tokens[1:],dtype=float)
# Build the hin embedding matrix
for hin_node_id in hin_vocabulary:
    hin_embeddings[hin_node_id] = node2vec[str(hin_node_id)]


# Prepare train test data
# Max length of summaries
max_seq_length = 5000
# max_seq_length = max(data_df.summary.map(lambda x: len(x)).max(),
#                      data_df.description.map(lambda x: len(x)).max())
# print("max_seq_length:",max_seq_length)
hin_cols = ['hin']
X = data_df[hin_cols + text_cols]
Y = data_df['valid']

##### Define 5-fold cross validation test harness
# Fix random seed for reproducibility
seed = 7
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

cvscores_TEXT = []
cvscores_HIN = []
cvscores_TEXT_HIN = []

for train, test in kfold.split(X, Y):
    MODEL_NO = str(MODEL_NO)
    print('\n*********** ' + 'Running 5-fold validation: '  + MODEL_NO + ' ! ***********\n')


    # Split to dicts -- train
    X_text_train = X.iloc[train].summary + X.iloc[train].description
    X_hin_train = X.iloc[train].hin

    # Split to dicts -- test
    X_text_test = X.iloc[test].summary + X.iloc[test].description
    X_hin_test =X.iloc[test].hin
    # Convert labels to their numpy representations
    Y_train = Y.iloc[train].values
    Y_test = Y.iloc[test].values

    # Zero padding
    for dataset in itertools.product([X_text_train, X_text_test]):
        dataset = pad_sequences(dataset, maxlen=max_seq_length)

    for dataset in itertools.product([X_hin_train, X_hin_test]):
        dataset = pad_sequences(dataset, maxlen=4)

    ##### Performing oversampling in the training sets at each iteration of 5-fold cross-validation procedure #####
    # Concatenate X_hin_train_left, X_hin_train_right, X_text_train_left, X_text_train_right
    X_train_cat = np.concatenate([X_hin_train,X_text_train],axis=-1)

    # Over-sampling using SMOTE and cleaning using Tomek links
    smt = SMOTETomek(random_state=42,n_jobs=14)
    X_train_cat_res, Y_train_res = smt.fit_resample(X_train_cat, Y_train)

    hin_features_num = np.shape(X_hin_train)[1]
    text_features_num = np.shape(X_text_train)[1]

    X_hin_train_res = X_train_cat_res[:,0:hin_features_num]
    X_text_train_res = X_train_cat_res[:,hin_features_num:hin_features_num+text_features_num]


    # Make sure everything is ok
    assert len(X_text_train_res) == len(Y_train_res)


    # Model variables
    n_hidden_rnn = 100
    n_dense_hin = 32
    n_dense_fusion = 64
    gradient_clipping_norm = 1.25
    batch_size = 128
    n_epoch = 100

    #####################
    ### I: Model Text (Text) ###
    MODEL_NAME = 'TEXT'
    K.clear_session()
    #####################

    ## Text Information Representation ##
    # 1) Text Input Layer
    bug_text_input = Input(shape=(max_seq_length,), dtype='int32', name='text_input')
    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim=len(word_embeddings),
                                output_dim=word_embedding_dim,
                                weights=[word_embeddings],
                                input_length=max_seq_length,
                                trainable=False,
                                name='text_embedding')
    bug_text_embedding = embedding_layer(bug_text_input)

    # 3) Shared Bi-LSTM
    shared_bilstm = Bidirectional(CuDNNLSTM(n_hidden_rnn, return_sequences=False, name='shared_bilstm'))
    bug_text_bilstm = shared_bilstm(bug_text_embedding)
    bug_text_repr = Dropout(0.25)(bug_text_bilstm)

    ## Malstm Distance Layer ##
    mahanttan_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)))
    mahanttan_distance_text = mahanttan_layer([bug_text_repr])

    ## Build the model ##
    model_text = Model(inputs=[bug_text_input], outputs=[mahanttan_distance_text])
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    model_text.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model_text.summary()

    ## Train the model ##
    training_start_time = time()
    model_trained = model_text.fit([X_text_train_res], Y_train_res, batch_size=batch_size,
                                   epochs=n_epoch, validation_split=0.2, shuffle=True)
    print("Training time finished.\n{} epochs in {}".format(n_epoch,
                                                            datetime.timedelta(seconds=time() - training_start_time)))

    ## Test ##
    Y_pred_text = model_text.predict([X_text_test])

    accuracy = accuracy_score(Y_test == 1, Y_pred_text >= 0.5)
    precision = precision_score(Y_test == 1, Y_pred_text >= 0.5, average=None)
    recall = recall_score(Y_test == 1, Y_pred_text >= 0.5, average=None)
    f_measure = f1_score(Y_test == 1, Y_pred_text >= 0.5, average=None)

    print("model_text: test accuracy: {0}".format(accuracy))
    print("model_text: test precision p: {0}".format(precision[1]))
    print("model_text: test recall p: {0}".format(recall[1]))
    print("model_text: test f1 score p: {0}".format(f_measure[1]))
    print("model_text: test precision n: {0}".format(precision[0]))
    print("model_text: test recall n: {0}".format(recall[0]))
    print("model_text: test f1 score n: {0}".format(f_measure[0]))

    cvscores_TEXT.append(accuracy * 100)
    ## Save model ##
    model_text.save(MODEL_SAVE_FILE + MODEL_NAME + '_' + MODEL_NO + '.h5')

    ## Record test result ##
    with open(EXP_TEST_HISTORY_FILE + MODEL_NAME, 'a') as f:
        f.write(str(accuracy) + '\t')
        f.write(str(precision[1]) + '\t')
        f.write(str(recall[1]) + '\t')
        f.write(str(f_measure[1]) + '\t')
        f.write(str(precision[0]) + '\t')
        f.write(str(recall[0]) + '\t')
        f.write(str(f_measure[0]) + '\n')

    ## Recore training history ##
    # Accuracy score
    with open(EXP_HISTORY_ACC_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['accuracy']:
            f.write(str(i) + '\t')
        f.write('\n')

    with open(EXP_HISTORY_VAL_ACC_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['val_accuracy']:
            f.write(str(i) + '\t')
        f.write('\n')

    # Loss
    with open(EXP_HISTORY_LOSS_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['loss']:
            f.write(str(i) + '\t')
        f.write('\n')

    with open(EXP_HISTORY_VAL_LOSS_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['val_loss']:
            f.write(str(i) + '\t')
        f.write('\n')

    del model_text
    del model_trained


    #################################
    ### III: Model Text Hin Dense (HIN2) ###
    MODEL_NAME = 'TEXT_HIN_DENSE'
    K.clear_session()
    #################################

    ## Text Information Representation ##
    # 1) Text Input Layer
    bug_text_input = Input(shape=(max_seq_length,), dtype='int32', name='text_input')
    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim = len(word_embeddings), 
                                output_dim = word_embedding_dim, 
                                weights=[word_embeddings], 
                                input_length=max_seq_length, 
                                trainable=False,
                                name='text_embedding')
    bug_text_embedding_left = embedding_layer(bug_text_input)
    # 3) Shared Bi-LSTM 
    shared_bilstm = Bidirectional(CuDNNLSTM(n_hidden_rnn, return_sequences=False, name='shared_bilstm'))
    bug_text_bilstm = shared_bilstm(bug_text_embedding)
    bug_text_repr = Dropout(0.25)(bug_text_bilstm)

    ## Hin Information Representation ##
    # 1) Hin Input Layer
    bug_hin_input = Input(shape=(6,), dtype='int32', name='hin_input')

    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim = len(hin_embeddings),
                            output_dim = hin_embedding_dim,
                            weights=[hin_embeddings],
                            input_length=6,
                            trainable=False,
                            name='hin_embedding')
    bug_hin_embedding = embedding_layer(bug_hin_input)
    bug_hin_embedding_flat = Flatten()(bug_hin_embedding)
    dense_layer = Dense(n_dense_hin, activation='tanh')
    bug_hin_repr = dense_layer(bug_hin_embedding_flat)

    ## Bug Report Representation ##
    merge_bug_text_hin = concatenate([bug_text_repr, bug_hin_repr])
    dense_layer = Dense(n_dense_fusion,activation='tanh',name='dense_bugrepr')
    bug_repr = dense_layer(merge_bug_text_hin)

    ## Malstm Distance Layer ##
    mahanttan_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
    mahanttan_distance_text_hin_dense = mahanttan_layer([bug_repr])
        
    ## Build the model ##
    model_text_hin_dense = Model(inputs=[bug_text_input, bug_hin_input], outputs=[mahanttan_distance_text_hin_dense])
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    model_text_hin_dense.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model_text_hin_dense.summary()

    ## Train the model ##
    training_start_time = time()
    model_trained = model_text_hin_dense.fit([X_text_train_res, X_hin_train_res], Y_train_res, batch_size=batch_size, epochs=n_epoch, validation_split=0.2, shuffle=True)

    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


    ## Test ##
    Y_pred_text_hin_dense = model_text_hin_dense.predict([X_text_test, X_hin_test])



    accuracy = accuracy_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5)
    precision = precision_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5, average=None)
    recall = recall_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5, average=None)
    f_measure = f1_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5, average=None)
    
    print("model_text_hin_dense: test accuracy: {0}".format(accuracy))
    print("model_text_hin_dense: test precision p: {0}".format(precision[1]))
    print("model_text_hin_dense: test recall p: {0}".format(recall[1]))
    print("model_text_hin_dense: test f1 score p: {0}".format(f_measure[1]))
    print("model_text_hin_dense: test precision n: {0}".format(precision[0]))
    print("model_text_hin_dense: test recall n: {0}".format(recall[0]))
    print("model_text_hin_dense: test f1 score n: {0}".format(f_measure[0]))

    ## Record test result ## 
    with open(EXP_TEST_HISTORY_FILE + MODEL_NAME, 'a') as f:
        f.write(str(accuracy) + '\t')
        f.write(str(precision[1]) + '\t')
        f.write(str(recall[1]) + '\t')
        f.write(str(f_measure[1]) + '\t')
        f.write(str(precision[0]) + '\t')
        f.write(str(recall[0]) + '\t')
        f.write(str(f_measure[0]) + '\n')

    cvscores_TEXT_HIN.append(accuracy * 100)

    ## Save model ## 
    model_text_hin_dense.save(MODEL_SAVE_FILE + MODEL_NAME + '_' + MODEL_NO + '.h5')


    ## Record training history ##
    # Accuracy score
    with open(EXP_HISTORY_ACC_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['accuracy']:
            f.write(str(i) + '\t')
        f.write('\n')

    with open(EXP_HISTORY_VAL_ACC_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['val_accuracy']:
            f.write(str(i) + '\t') 
        f.write('\n')

    # Loss
    with open(EXP_HISTORY_LOSS_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['loss']:
            f.write(str(i) + '\t')
        f.write('\n')

    with open(EXP_HISTORY_VAL_LOSS_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['val_loss']:
            f.write(str(i) + '\t')
        f.write('\n')

    del model_text_hin_dense
    del model_trained

    MODEL_NO = int(MODEL_NO)
    MODEL_NO += 1


