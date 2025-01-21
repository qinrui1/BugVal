import tensorflow as tf
from keras import layers
import numpy as np
import pandas as pd
from keras.layers import Input, Embedding, Layer, Dropout


class Embedding(Layer):
    def __init__(self,vocab_size,model_dim,**kwargs):
        self.vocab_size=vocab_size
        self.model_dim=model_dim
        super(Embedding,self).__init__(**kwargs)
    def build(self,input_shape):
        self.embeddings=self.add_weight(
            shape=(self.vocab_size,self.model_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="embeddings"
        )
        super(Embedding,self).build(input_shape)
    def call(self,inputs):#其实就是简单的取个行向量出来
        inputs=tf.cast(inputs,tf.int32)
        embeddings=tf.gather(self.embeddings,inputs)
        embeddings*=self.model_dim**0.5 #暂时不清楚为什么
        return embeddings
    def get_config(self):
        config=super(Embedding,self).get_config()
        config.update({
            "vocab_size":self.vocab_size,
            "model_dim":self.model_dim
        })
        return config
class PositionEncoding(Layer):
    def __init__(self,**kwargs):
        super(PositionEncoding,self).__init__(**kwargs)
    def build(self,input_shape):
        def get_position_encoding(seq_len,model_dim):
            position_encoding=np.zeros(shape=(seq_len,model_dim))
            for pos in range(seq_len):
                for i in range(model_dim):
                    position_encoding[pos,i]=pos/(np.power(10000,2*i/model_dim))
            position_encoding[::,::2]=np.sin(position_encoding[::,::2])
            position_encoding[::,1::2]=np.cos(position_encoding[::,1::2])
            return np.expand_dims(position_encoding,axis=0)
        seq_len,model_dim=input_shape.as_list()[1:3]
        self.position_encoding=self.add_weight(
            shape=(1,seq_len,model_dim),
            initializer=tf.constant(get_position_encoding(seq_len,model_dim)),
            trainable=False,
            name="position_encoding"
        )
        super(PositionEncoding,self).build(input_shape)
    def call(self,inputs):
        return self.position_encoding
def masks(self,inputs,masking):
    masking=tf.cast(masking,tf.float32)
    masking=tf.tile(masking,[tf.shape(inputs)[0]//tf.shape(masking)[0],1])
    #因为MultiHeadAttention的问题,masking的长度和inputs
    #长度可能并不等价，而是倍数关系，具体为n_head倍
    masking=tf.expand_dims(masking,axis=1)
    outputs=inputs+masking*self.masking_num
    #乘以一个很大的负数，目的是为了让当前位置的数值失效
    return outputs

def lookahead_mask(self, inputs):  # 前瞻遮挡,上三角矩阵masks
    diag_masks = 1 - tf.linalg.band_part(tf.ones_like(inputs), -1, 0)
    paddings = tf.ones_like(inputs) * self.masking_num
    outputs = tf.where(tf.equal(diag_masks, 0), inputs, paddings)  # 经过softmax,outputs变为下三角矩阵
    return outputs

class ScaledDotProductAttention(Layer):
    def __init__(self,masking=True,lookahead_masking=False,dropout_rate=0,**kwargs):
        self.masking=masking
        self.lookahead_masking=lookahead_masking
        self.dropout_rate=dropout_rate
        self.masking_num=-1e9
        super(ScaledDotProductAttention,self).__init__(**kwargs)
    def masks(self,inputs,masking):
        masking=tf.cast(masking,tf.float32)
        masking=tf.tile(masking,[tf.shape(inputs)[0]//tf.shape(masking)[0],1])
        masking=tf.expand_dims(masking,axis=1)
        outputs=inputs+masking*self.masking_num
        return inputs
    def lookahead_masks(self,inputs):
        ones=tf.ones_like(inputs)
        diag_masking=1-tf.linalg.band_part(inputs,num_lower=-1,num_upper=0)
        paddings=ones*self.masking_num
        outputs=tf.where(tf.equal(diag_masking,0),inputs,paddings)
        return outputs
    def call(self,inputs):
        if self.masking:
            queries,keys,values,masking=inputs
        else:
            queries,keys,values=inputs
        model_dim=queries.shape.as_list()[-1]
        matmul=tf.matmul(queries,tf.transpose(keys,[0,2,1]))
        scaled=matmul/model_dim**0.5
        if self.masking:
            scaled=self.masks(scaled,masking)
        if self.lookahead_masking:
            scaled=self.lookahead_masks(scaled)
        softmax=tf.nn.softmax(scaled)
        softmax=Dropout(self.dropout_rate)(softmax)
        outputs=tf.matmul(softmax,values)
        return outputs
    def get_config(self):
        config=super(ScaledDotProductAttention.self).get_config()
        config.update({
            "masking":self.masking,
            "lookahead_masking":self.lookahead_masking,
            "dropout_rate":self.dropout_rate,
            "masking_num":self.masking_num
        })
        return config

class MultiHeadAttention(Layer):
    def __init__(self,n_head=8,head_dim=64,dropout_rate=0.1,masking=True,lookahead_masking=False,trainable=True,**kwargs):
        self.n_head=n_head
        self.head_dim=head_dim
        self.dropout_rate=dropout_rate
        self.masking=masking
        self.lookahead_masking=lookahead_masking
        self.trainable=trainable
        super(MultiHeadAttention,self).__init__(**kwargs)
    def build(self,input_shape):
        self.queries_weight=self.add_weight(
            shape=(input_shape[0][-1],self.head_dim*self.n_head),
            initializer="glorot_uniform",
            trainable=self.trainable,
            name="queries_weight",
        )
        self.keys_weight=self.add_weight(
            shape=(input_shape[0][-1],self.head_dim*self.n_head),
            initializer="glorot_uniform",
            trainable=self.trainable,
            name="keys_weight"
        )
        self.values_weight=self.add_weight(
            shape=(input_shape[0][-1],self.head_dim*self.n_head),
            initializer="glorot_uniform",
            trainable=self.trainable,
            name="values_weight"
        )
        super(MultiHeadAttention,self).build(input_shape)
    def call(self,inputs):
        if self.masking:
            queries,keys,values,masks=inputs
        else:
            queries,keys,values=inputs
        queries=tf.matmul(queries,self.queries_weight)
        keys=tf.matmul(keys,self.keys_weight)
        values=tf.matmul(values,self.values_weight)
        queries=tf.concat(tf.split(queries,self.n_head,axis=-1),axis=0)
        keys=tf.concat(tf.split(keys,self.n_head,axis=-1),axis=0)
        values=tf.concat(tf.split(values,self.n_head,axis=-1),axis=0)
        if self.masking:
            attention_input=[queries,keys,values,masks]
        else:
            attention_input=[queries,keys,values]
        attention=ScaledDotProductAttention(
            masking=self.masking,
            lookahead_masking=self.lookahead_masking,
            dropout_rate=self.dropout_rate,
        )
        attention_out=attention(attention_input)
        outputs=tf.concat(tf.split(attention_out,self.n_head,axis=0),axis=-1)
        return outputs
    def get_config(self):
        config=super(ScaledDotProductAttention,self).get_config()
        config.update({
            "n_head":self.n_head,
            "head_dim":self.head_dim,
            "dropout_rate":self.dropout_rate,
            "masking":self.masking,
            "lookahead_masking":self.lookahead_masking,
            "trainable":self.trainable
        })
        return config


