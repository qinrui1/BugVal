import pickle

# rb是2进制编码文件，文本文件用r
f1 = open(r'../model/word2vec/thunderbird/exp_word_index.pkl','rb')
data1 = pickle.load(f1)
len1 = len(data1)
print('word_index：',data1)
print('\n word_index length：',len1)
f2 = open(r'../model/word2vec/thunderbird/exp_embeddings_matrix.pkl','rb')
data2 = pickle.load(f2)
len2 = len(data2)
print('\n embeddings_matrix：',data2)
print('\n word_index length：',len2)