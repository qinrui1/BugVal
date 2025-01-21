import warnings
warnings.filterwarnings('ignore')
from gensim.models import word2vec
import gensim
import numpy as np
import pickle
import pandas as pd

class MySentences(object):
    def __init__(self, sentences):
        self.sentences = sentences


    def __iter__(self):
        for text in self.sentences:
            yield text.split()
              
class W2v:
    def __init__(self):
        self.save_path = '../model/word2vec/mozilla/'
        self.csv_path = '../data/mozilla/mozilla_after_nltk1.csv'
        
    def load_file(self):
        dataFrame = pd.read_csv(self.csv_path,encoding='utf-8')
        texts_summary = []   # 存储读取的 x
        texts_description = []
        labels = []  # 存储读取的y
        # 遍历 获取数据
        for i in range(len(dataFrame)):
            texts_summary.append(dataFrame.at[i,'summary']) # 每个元素为一句话“《机械设计基础》这本书的作者是谁？”
            texts_description.append(dataFrame.at[i,'description'])
            labels.append(dataFrame.at[i,'valid']) # 每个元素为一个int 代表类别 # [2, 6, ... 3] 的形式。减一为了 从0开始
            
        # 把类别从int 3 转换为(0,0,0,1,0,0)的形式
        #labels = to_categorical(np.asarray(labels)) # keras的处理方法，一定要学会# 此时为[[0. 0. 1. 0. 0. 0. 0.]....] 的形式
        texts=[]
        for text in zip(texts_summary,texts_description):
            texts.append(text)
        return texts, labels # 总文本，总标签
    
    def list_split(self, list, groups):
        len_list = int(len(list))+1
        size = int(len_list/groups)+1
        result=[]
        for i in range(0,len_list,size):
            item = list[i:i+size]
            result.append(item)
        return result
    
    def split_data_ordered(self, texts, labels):
        texts_list = self.list_split(texts,11)
        labels_list = self.list_split(labels,11)
        summary_list=[]
        description_list=[]
        for texts in texts_list:
            summary=[]
            description=[]
            for text in texts:
                summary.append(text[0])
                description.append(text[1])
            summary_list.append(summary)
            description_list.append(description)
        return summary_list, description_list, labels_list
    
    def lists_merge(self, lists):
        list_merge=[]
        for list in lists:
            list_merge = list_merge + list
        return list_merge
        
    def train(self,sentences):
        sentences = MySentences(sentences)
        model = word2vec.Word2Vec(sentences, vector_size=128, window=5, min_count=5, workers=4, sg=1)
        model.save(self.save_path+'word2vec.model')
        return model
        
    def test(self):
        model = gensim.models.Word2Vec.load(self.save_path+'word2vec.model')
        print(model)
        print('打印与attachment最相近的10个词语：',model.most_similar('attachment', topn=10))
        
    def save_obj(self, obj, target, name):
        pickle.dump(obj, open(target+name, 'wb'),protocol=4)
        
    def construct_dic(self,model):
        word_index = {" ": 0}
        word_vector = {}
        embeddings_matrix=[]
        vocab_list = list(model.wv.index_to_key)

        embeddings_matrix = np.zeros((len(vocab_list) + 1, model.vector_size))
        for i in range(len(vocab_list)):
            word = vocab_list[i]  # 每个词语
            word_index[word] = i + 1 # 词语：序号
            word_vector[word] = model.wv[word] # 词语：词向量
            embeddings_matrix[i + 1] = model.wv[word]
            
        self.save_obj(word_index,self.save_path,'word_index.pkl')
        self.save_obj(embeddings_matrix,self.save_path,'embeddings_matrix.pkl')
        return word_index, word_vector, embeddings_matrix
    
if __name__ == '__main__':

    w2v=W2v()
    texts, labels = w2v.load_file()
    summary_list, description_list, label_list = w2v.split_data_ordered(texts, labels)
    summary=w2v.lists_merge(summary_list[0:10])
    description=w2v.lists_merge(description_list[0:10])
    sentence=[]
    sentence.append(summary)
    sentence.append(description)
    sentences=w2v.lists_merge(sentence)
    sentences=[str(text) for text in sentences]
    model=w2v.train(sentences)
    w2v.construct_dic(model)
    '''
    w2v.test()
    '''