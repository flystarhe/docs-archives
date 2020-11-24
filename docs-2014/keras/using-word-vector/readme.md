title: 在Keras模型中使用预训练的词向量
date: 2017-07-19
tags: [Keras,词向量,文本分类]
---
通过本教程,你可以掌握技能:使用预先训练的词向量和卷积神经网络解决一个文本分类问题.

<!--more-->
## 什么是词向量?

"词向量"(词嵌入)是一类将词的语义映射到向量空间中去的自然语言处理技术.即将一个词用特定的向量来表示,向量之间的距离(例如,任意两个向量之间的L2范式距离或更常用的余弦距离)一定程度上表征了的词之间的语义关系.由这些向量形成的几何空间被称为一个嵌入空间.词向量通过降维技术表征文本数据集中的词的共现信息,方法包括神经网络(Word2vec技术),或矩阵分解.

## GloVe词向量

本文使用[GloVe词向量](http://nlp.stanford.edu/projects/glove/).GloVe是"Global Vectors for Word Representation"的缩写,一种基于共现矩阵分解的词向量.本文所使用的GloVe词向量是在2014年的英文维基百科上训练的,有400k个不同的词,每个词用100维向量表示.[点此下载](http://nlp.stanford.edu/data/glove.6B.zip),词向量文件约为822M.

## 20 Newsgroup dataset

本文使用的数据集是著名的"20 Newsgroup dataset".该数据集共有20种新闻文本数据,我们将实现对该数据集的文本分类任务.数据集的说明和下载请参考[这里](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html).

## 实验方法

以下是我们如何解决分类问题的步骤:

- 将所有的新闻样本转化为词索引序列,只保留最参见的2w个词,每个新闻最多保留1k个词
- 生成一个词向量矩阵.第i列表示词索引为i的词的词向量
- 将词向量矩阵载入Keras Embedding层,设置该层的权重不可再训练
- Keras Embedding层之后连接一个1D的卷积层,并用一个softmax全连接输出新闻类别

## 数据预处理

首先遍历语料文件下的所有文件夹,获得不同类别的新闻以及对应的类别标签,代码如下:
```
BASE_DIR = '/data1/hejian_lab/_temp/news20/'
TEXT_DATA_DIR = BASE_DIR + '20_newsgroup/'

import os

labels_index = {}  # dictionary mapping label name to numeric id
texts = []  # list of text samples
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath, encoding='latin-1')
                texts.append(f.read())  # you need skip header
                f.close()
                labels.append(label_id)

print('Found %d texts %d class.' % (len(texts), len(labels_index)))

import pandas as pd
print(pd.Series(labels).value_counts())
```

之后,将新闻样本转化为神经网络训练所用的张量.所用到的Keras库是`keras.preprocessing.text.Tokenizer`和`keras.preprocessing.sequence.pad_sequences`.代码如下:
```
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
VALIDATION_SPLIT = 0.2

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)  # 训练的文本列表
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index  # 从1开始编号
print('Found %d unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre')

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
```

`pad_sequences`效果如下:
```
pad_sequences([[1,2,3,4],[1,2,3],[1,2],[1],[]], maxlen=3)
## array([[2, 3, 4],
##        [1, 2, 3],
##        [0, 1, 2],
##        [0, 0, 1],
##        [0, 0, 0]], dtype=int32)
```

`to_categorical`效果如下:
```
to_categorical([1,2,3,1,4])
## array([[ 0.,  1.,  0.,  0.,  0.],
##        [ 0.,  0.,  1.,  0.,  0.],
##        [ 0.,  0.,  0.,  1.,  0.],
##        [ 0.,  1.,  0.,  0.,  0.],
##        [ 0.,  0.,  0.,  0.,  1.]])
```

## Embedding layer设置
接下来,我们从GloVe文件中解析出每个词和它所对应的词向量,并用字典的方式存储:
```
GLOVE_DIR = BASE_DIR + 'glove.6B/'

import os
import numpy as np

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %d word vectors.' % len(embeddings_index))
```

此时,我们可以根据得到的字典生成上文所定义的词向量矩阵:
```
EMBEDDING_DIM = 100  # 因为 embeddings_index 使用 glove.6B.100d 初始化

import numpy as np

num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    # 注意,i是从1开始的
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
```

现在我们将这个词向量矩阵加载到Embedding层中,设置`trainable=False`使得这个编码层不可再训练:
```
from keras.layers import Embedding

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
```

一个Embedding层的输入应该是一系列的整数序列,比如一个2D的输入,它的shape值为`(samples,indices)`,也就是一个samples行,indeces列的矩阵.每一次的batch训练的输入应该被padded成相同大小(尽管Embedding层有能力处理不定长序列,如果你不指定数列长度这一参数).所有的序列中的整数都将被对应的词向量矩阵中对应的列(也就是它的词向量)代替,比如序列`[1,2]`将被序列`[词向量_1,词向量_2]`代替.这样,输入一个2D张量后,我们可以得到一个3D张量.

## 训练1D卷积
最后,使用一个小型的1D卷积解决这个新闻分类问题:
```
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
hist = model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
print(['%s: %.4f' % (key, val[-1]) for (key, val) in hist.history.items()])
```

`2_epoch`达到0.94的分类准确率,`10_epoch`达到0.97,`4:1`分割训练和测试集合.你可以利用正则方法,如dropout,或在Embedding层上进行fine-tuning获得更高的准确率.做一个对比实验,直接使用Keras自带的Embedding层训练词向量而不用GloVe向量.代码如下:
```
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)
```

`2_epoch`之后,得到0.9的准确率.所以使用预训练的词向量作为特征是非常有效的.一般来说,在自然语言处理任务中,当样本数量非常少时,使用预训练的词向量是可行的.实际上,预训练的词向量引入了外部语义信息,往往对模型很有帮助.

## 参考资料:
- [中文/在Keras模型中使用预训练的词向量](http://keras-cn.readthedocs.io/en/latest/blog/word_embedding/)
- [英文/Using pre-trained word embeddings in a Keras model](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
