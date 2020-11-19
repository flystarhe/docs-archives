title: 中文维基百科语料上的Word2Vec实验
date: 2016-09-04
tags: [NLP,Word2Vec,Gensim,GloVe]
---
自然语言理解的问题要转化为机器学习的问题，第一步肯定是要找一种方法把这些符号数学化。NLP中最直观，也是到目前为止最常用的词表示方法是`One-hot`，这种表示方法一个重要的问题就是“词汇鸿沟”现象。word2vec具体方式有两种，一种是用上下文预测目标词（连续词袋法，简称CBOW），另一种则是用一个词来预测一段目标上下文，称为skip-gram方法。skip-gram在处理大规模数据集时结果更为准确。

<!--more-->
## word2vec思想
Word2vec实际上是两种不同的方法：Continuous Bag of Words(CBOW)和Skip-gram。CBOW的目标是根据上下文来预测当前词语的概率。Skip-gram刚好相反，根据当前词语来预测上下文的概率。这两种方法都利用人工神经网络作为它们的分类算法。起初，每个单词都是一个随机N维向量。训练时，该算法利用CBOW或者Skip-gram的方法获得了每个单词的最优向量。

Hierarchical Softmax用Huffman编码构造二叉树，其实借助了分类问题中，使用一连串二分类近似多分类的思想。例如我们是把所有的词都作为输出，那么“桔子”、“汽车”都是混在一起。给定`w_t`的上下文，先让模型判断`w_t`是不是名词，再判断是不是食物名，再判断是不是水果，再判断是不是“桔子”。在训练过程中，模型会赋予这些抽象的中间结点一个合适的向量，这个向量代表了它对应的所有子结点。因为真正的单词公用了这些抽象结点的向量，所以Hierarchical Softmax方法和原始问题并不是等价的，但是这种近似并不会显著带来性能上的损失同时又使得模型的求解规模显著上升。

与潜在语义分析（Latent Semantic Index, LSI）、潜在狄立克雷分配（Latent Dirichlet Allocation，LDA）的经典过程相比，word2vec利用了词的上下文，语义信息更加地丰富。

## 安装gensim&jieba
这里以CentOS为例，其他系统环境可能需要稍作变化：

    $ wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
    $ bash Anaconda2-4.1.1-Linux-x86_64.sh
    $ conda update conda
    $ conda install numpy scipy gensim
    $ pip install jieba #尽量使用conda install安装

## 分词与预处理
该步骤包含分词，剔除标点符号和去文章结构标识。（建议word2vec训练数据不要去除标点符号，比如在情感分析应用中标点符号很有用）最终将得到分词好的纯文本文件，每行对应一篇文章，词语间以空格作为分隔符。`script_seg.py`如下：
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, codecs
import jieba.posseg as pseg

reload(sys)
sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python script.py infile outfile"
        sys.exit()
    i = 0
    infile, outfile = sys.argv[1:3]
    output = codecs.open(outfile, 'w', 'utf-8')
    with codecs.open(infile, 'r', 'utf-8') as myfile:
        for line in myfile:
            line = line.strip()
            if len(line) < 1:
                continue
            if line.startswith('<doc'):
                i = i + 1
                if(i % 1000 == 0):
                    print('Finished ' + str(i) + ' articles')
                continue
            if line.startswith('</doc'):
                output.write('\n')
                continue
            words = pseg.cut(line)
            for word, flag in words:
                if flag.startswith('x'):
                    continue
                output.write(word + ' ')
    output.close()
    print('Finished ' + str(i) + ' articles')
```

终端执行指令如下：

    $ time python script_seg.py std_zh_wiki_00 seg_std_zh_wiki_00
    $ time python script_seg.py std_zh_wiki_01 seg_std_zh_wiki_01
    $ time python script_seg.py std_zh_wiki_02 seg_std_zh_wiki_02

其中`std_zh_wiki_0*`来源于[维基百科中文语料的获取](https://flystarhe.github.io/2016/08/31/wiki-corpus-zh/)一文中的结果，我大概用了7小时。然后`$ cat seg_std_zh_wiki_00 seg_std_zh_wiki_01 seg_std_zh_wiki_02 >> zh.wiki`合并文件，然后再`$ sed '/^$/d' zh.wiki > zh.wiki.text`删除空白行，方便后续操作。

## 训练word2vec模型
这里选择的是Python版的[word2vec](http://radimrehurek.com/gensim/models/word2vec.html)。`script_train.py`如下：
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, codecs
import gensim, logging, multiprocessing

reload(sys)
sys.setdefaultencoding('utf-8')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python script.py infile outfile"
        sys.exit()
    infile, outfile = sys.argv[1:3]
    model = gensim.models.Word2Vec(gensim.models.word2vec.LineSentence(infile), size=400, window=5, min_count=5, sg=0, workers=multiprocessing.cpu_count())
    model.save(outfile)
    model.save_word2vec_format(outfile + '.vector', binary=False)
```

`min_count=5`指示内部字典修剪，在一个数十亿字语料中只有一次或两次出现的话，可能是无趣的错别字和垃圾，也没有足够的数据来做出那些话任何有意义的训练，所以最好忽略它们，一个合理的值是`(0,100)`，这取决于你的数据集的大小。默认`sg=0`表示使用CBOW训练算法，使用skip-gram训练算法则设置`sg=1`。终端执行`$ time python script_train.py zh.wiki.text zh.wiki.model`，大约要跑20多分钟（skip-gram效果好些，不过要跑1小时）。现在我们得到了一个gensim中默认格式的模型`zh.wiki.model`和一个原始c版本word2vec的vector格式的模型`zh.wiki.model.vector`。

## 使用word2vec模型
接下来通过gensim来加载和测试这个模型，因为`zh.wiki.model.vector`有2G，所以加载的时间也稍长一些：

    $ python
    >>> from gensim.models import Word2Vec
    >>> model = Word2Vec.load_word2vec_format("zh.wiki.model.vector", binary=False)
    >>> model[u"男人"] #词向量
    array([  3.70501429e-01,  -2.38224363e+00,  -1.20320223e-01,  ..
    >>> model.similarity(u"男人", u"女人")
    0.8284998105297946
    >>> print model.doesnt_match(u"早餐 晚餐 午餐 中心".split())
    中心
    >>> words = model.most_similar(u"男人")
    >>> for word in words:
    ...     print word[0], word[1]
    ... 
    女人 0.828499913216
    陌生人 0.639008104801
    女孩 0.635419011116
    女孩子 0.62975871563
    小女孩 0.608961224556
    男孩 0.599099874496
    小孩 0.597167491913
    小孩子 0.586811900139
    中年男人 0.57114815712
    家伙 0.569163799286
    >>> words = model.most_similar(u"女人")
    >>> for word in words:
    ...     print word[0], word[1]
    ... 
    男人 0.828499913216
    女孩 0.628861427307
    陌生人 0.614944934845
    小女孩 0.611582875252
    女孩子 0.608739495277
    寡妇 0.588962078094
    妓女 0.572488307953
    老婆 0.571004271507
    温柔 0.569867432117
    少妇 0.567982912064
    >>> words = model.most_similar(positive=[u"女人", u"皇后"], negative=[u"男人"])
    >>> for word in words:
    ...     print word[0], word[1]
    ... 
    皇太后 0.59976452589
    太后 0.577173352242
    妃 0.554282069206
    太皇太后 0.551343083382
    王后 0.548985362053
    王妃 0.540383219719
    妃子 0.536857843399
    元妃 0.535273551941
    贵妃 0.533002495766
    贵人 0.527253031731
    >>> model.n_similarity([u"女人", u"皇帝"], [u"男人", u"皇后"])
    0.73030644595280991

## 评估word2vec模型
Word2vec训练是一种无监督的任务，没有什么好的办法客观评价结果。评价取决于你的应用。谷歌已经发布了测试集，约20000句法和语义的测试实例，检查如`A对于B类似C对于D`这种线性平移关系。我考虑翻译[questions-words.txt](https://code.google.com/archive/p/word2vec/source/default/source)为中文来测试。`script_trans.py`如下：
```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys, codecs, re, urllib, urllib2

reload(sys)
sys.setdefaultencoding('utf-8')

def translate(text, f, t):
    values = {'hl':'zh-CN','ie':'UTF-8','text':text,'langpair':"%s|%s"%(f,t)}
    url = 'https://translate.google.com'
    data = urllib.urlencode(values)
    request = urllib2.Request(url, data)
    browser = 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 2.0.50727)'
    request.add_header('User-Agent', browser)
    response = urllib2.urlopen(request)
    html = response.read()
    p = re.compile(r"(?<=TRANSLATED_TEXT=').*?';")
    m = p.search(html)
    return m.group(0).strip("';")

def proc(infile, outfile):
    output = codecs.open(outfile, 'w', 'utf-8')
    with codecs.open(infile, 'r', 'utf-8') as myfile:
        for line in myfile:
            if line.startswith(':'):
                output.write(line)
                continue
            words = translate(line.strip().replace(' ', '/'), 'en', 'zh-CN')
            output.write(words.replace('/', ' ') + '\n')
    output.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: python script.py infile outfile"
        sys.exit()
    infile, outfile = sys.argv[1:3]
    proc(infile, outfile)
```

因为google翻译api收费，走`https://translate.google.com`又太慢，还要开代理。所以考虑把文件切碎，跑一分部做实验用：

    $ mkdir tmp
    $ split -l 1000 -d questions-words.txt tmp/
    $ cd tmp
    $ time python ../script_trans.py 00 zh_00
    $ time python ../script_trans.py 01 zh_01
    $ cat zh_00 zh_01 >> questions-words.zh
    $ sed '/^$/d' questions-words.zh > questions-words.zh.txt

终于得到`questions-words.zh.txt`文件，耗时76分钟。当然我们知道这样翻译过来的测试数据会有些问题，毕竟google是针对英文特性来构造的。但也只有凑合试试：

    >>> import gensim, logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    >>> model = gensim.models.Word2Vec.load("zh.wiki.model")
    >>> msgs = model.accuracy('questions-words.zh.txt')
    2016-09-07 18:22:13,833 : INFO : capital-common-countries: 75.0% (315/420)
    2016-09-07 18:22:14,315 : INFO : capital-world: 75.5% (71/94)
    2016-09-07 18:22:14,315 : INFO : total: 75.1% (386/514)
    >>> for msg in msgs:
    ...     for key, val in msg.items():
    ...         print 'key:',key
    ...         if isinstance(val, list):
    ...             for v in val:
    ...                 for i in v:
    ...                     print i,
    ...                 print
    ...         else:
    ...             print val

>在这个测试表现不错，并不代表word2vec会在你的应用程序工作的很好，反之亦然。它最好直接在您预期任务的中来评估。

## 继续训练word2vec
你可能希望加载一个存在的word2vec模型，然后使用更多的句子继续训练它：

    >>> model = gensim.models.Word2Vec.load("zh.wiki.model")
    >>> more_sentences = [[u'小剑', u'你好', u'啊'], [u'小剑', u'阳光', u'啊']]
    >>> model.similarity(u"你好", u"阳光")
    0.1648575233857773
    >>> model.train(more_sentences)
    >>> model.similarity(u"你好", u"阳光")
    0.16611509536924407
    >>> model.train(more_sentences)
    >>> model.similarity(u"你好", u"阳光")
    0.169120904998351

>这里只对从`model.save()`恢复的模型有效，从`model.save_word2vec_format()`恢复过来的模型只能用于查询。

## 使用GloVe训练词向量
GloVe也是无监督词向量学习算法。有网友评论：word2vec不开启负抽样时效果较GloVe略差，开启负抽样时效果较GloVe略好，不过word2vec开启负抽样训练过程会漫长很多，而GloVe在训练时更吃内存。感觉是时间换空间的选择，性能没有太大差别。安装GloVe：

    $ wget http://nlp.stanford.edu/software/GloVe-1.2.zip
    $ unzip GloVe-1.2.zip
    $ cd GloVe-1.2
    $ make
    gcc src/glove.c -o build/glove -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result
    cc1: error: invalid option argument ‘-Ofast’
    cc1: warning: unrecognized command line option "-Wno-unused-result"
    $ vim Makefile #这时你需要修改Makefile或升级gcc
    CFLAGS = -lm -pthread -Ofast -march=native -funroll-loops -Wno-unused-result
    #替换为：阅读第二行说明
    CFLAGS = -lm -pthread -O2 -march=native -funroll-loops -Wno-unused-result
    $ make
    $ ./demo.sh

`demo.sh`的工作分3部分：检查有没有`text8`语料，如果没有就自己去网上下；展示训练步骤`$BUILDDIR/vocab_count -> $BUILDDIR/cooccur -> $BUILDDIR/shuffle -> $BUILDDIR/glove`；对模型做评价`python eval/python/evaluate.py`。较长的等待后你会得到以下文件：

- vocab_count#从语料库中统计词频，输出文件`vocab.txt`，每行为`词语 词频`；
- cooccur#从语料库中统计词共现，输出文件`cooccurrence.bin`，二进制文件；
- shuffle#对`cooccurrence.bin`重新整理，输出文件`cooccurrence.shuf.bin`；
- glove#训练模型，输出文件`vectors.bin + vectors.txt`；

`vectors.txt`与`zh.wiki.model.vector`的区别是后者多了个`554353 400`，第一个数是词语的数量，第二个数是词向量大小。所以把`vectors.txt`给gensim用，只需在文件前端插入：

    $ wc -l vectors.txt #获取行数/词数量
    71291 vectors.txt
    $ sed '1i 71291 50' vectors.txt > vectors.txt.word2vec #50是词向量大小

对GloVe有了基本了解，现在准备用前文的`zh.wiki.text`语料训练模型，然后用`questions-words.zh.txt`测试并与gensim训练效果对比：

    $ time build/vocab_count -min-count 5 -verbose 2 < zh.wiki.text > zh.wiki.vocab

耗时约30秒，其中`-min-count 5`指示词频低于5的词舍弃，`-verbose 2`控制屏幕打印信息的，不想看就设为0。

    $ time build/cooccur -memory 4.0 -vocab-file zh.wiki.vocab -verbose 2 -window-size 5 < zh.wiki.text > zh.wiki.cooccurrence.bin

耗时约23分钟，其中`-memory 4.0`指示`bigram_table`缓冲器，`-vocab-file zh.wiki.vocab`不需要解释，`-verbose 2`同上，`-window-size 5`指示词窗口大小。

    $ time build/shuffle -memory 4.0 -verbose 2 < zh.wiki.cooccurrence.bin > zh.wiki.cooccurrence.shuf.bin

耗时约4分钟，终于可以训练词向量了：

    $ time build/glove -save-file zh.wiki.vectors.glove -threads 8 -input-file zh.wiki.cooccurrence.shuf.bin -vocab-file zh.wiki.vocab -x-max 10 -iter 5 -vector-size 400 -binary 0 -verbose 2

耗时约1小时(比gensim慢嘛，这还没计前面两步的时间，都是骗子)，其中`-save-file zh.wiki.vectors.glove`、`-threads 8`、`-input-file zh.wiki.cooccurrence.shuf.bin`和`-vocab-file zh.wiki.vocab`按字面理解就对了，`-x-max 10`没读懂，`-iter 5`迭代5轮，`-vector-size 400`词向量400维，`-binary 0`控制输出格式`0: save as text files; 1: save as binary; 2: both. For binary`。有些参数我故意设低，不是没耐心等待，而是想尽量和gensim训练word2vec参数相当，以便于对比。是时候看成效了：

    $ wc -l zh.wiki.vectors.glove.txt #获取行数/词数量
    554354 zh.wiki.vectors.glove.txt
    $ sed '1i 554354 400' zh.wiki.vectors.glove.txt > zh.wiki.model.glove
    $ python
    >>> import gensim, logging
    >>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    >>> model = gensim.models.Word2Vec.load_word2vec_format("zh.wiki.model.glove", binary=False)
    >>> msgs = model.accuracy('questions-words.zh.txt')
    2016-09-08 16:08:20,061 : INFO : capital-common-countries: 82.1% (345/420)
    2016-09-08 16:08:20,569 : INFO : capital-world: 80.9% (76/94)
    2016-09-08 16:08:20,570 : INFO : total: 81.9% (421/514)

从测试集表现来看，比gensim的75%提升了5个点，不过训练过程真心漫长。个人感觉GloVe比gensim慢很多，不过效果确实有些提升。

## 参考资料：
- [Word2vec Tutorial](http://rare-technologies.com/word2vec-tutorial/)
- [中英文维基百科语料上的Word2Vec实验](http://www.52nlp.cn/中英文维基百科语料上的Word2Vec实验)
- [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/projects/glove/)
- [Word2vec: Neural Word Embeddings in Java](http://deeplearning4j.org/zh-word2vec)
- [Google公开的英文word2vec模型](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)