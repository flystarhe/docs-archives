title: 如何计算两个文档的相似度
date: 2016-08-30
tags: [驾驭文本,文档相似度,Ubuntu]
---
思路很简单，就是将文本内容映射到topic的维度，然后再计算其相似度。topic模型采用LSI(Latent semantic indexing:浅层语义索引)。最后是gensim这个强大的Python工具包。

<!--more-->
## TF-IDF，余弦相似度，向量空间模型
这几个知识点在信息检索中是最基本的，入门级的参考资料可以看看吴军老师在《数学之美》中第11章“如何确定网页和查询的相关性”和第14章“余弦定理和新闻的分类”中的通俗介绍或者阮一峰老师写的两篇科普文章[TF-IDF与余弦相似性的应用（一）：自动提取关键词](http://www.ruanyifeng.com/blog/2013/03/tf-idf.html)和[TF-IDF与余弦相似性的应用（二）：找出相似文章](http://www.ruanyifeng.com/blog/2013/03/cosine_similarity.html)。

## SVD和LSI
想了解LSI一定要知道SVD(Singular value decomposition:奇异值分解)，而SVD的作用不仅仅局限于LSI，在很多地方都能见到其身影，SVD自诞生之后，其应用领域不断被发掘，可以不夸张的说如果学了线性代数而不明白SVD，基本上等于没学。推荐MIT教授`Gilbert Strang`的线性代数公开课和相关书籍，你可以直接在网易公开课看相关章节的视频。

一种情况下我们考察两个词的关系常常考虑的是它们在一个窗口长度（譬如一句话，一段话或一个文章）里的共现情况，在语料库语言学里有个专业点叫法叫Collocation(词语搭配)。而LSI所做的是挖掘如下这层词语关系：A和C共现，B和C共现，目标是找到A和B的隐含关系，学术一点的叫法是`second-order co-ocurrence`。

## LDA
推荐rickjin的“LDA数学八卦”系列，通俗易懂，娓娓道来，另外rick的其他系列也是非常值得一读的。

## 安装gensim
gensim依赖NumPy和SciPy这两大Python科学计算工具包，一种简单的安装方法是`pip install`：
```
$ sudo apt-get install unzip python python-dev python-pip
$ sudo apt-get install python-numpy python-scipy
$ sudo pip install --upgrade gensim
```

## 使用gensim
gensim的官方[tutorial](http://radimrehurek.com/gensim/tutorial.html)非常详细，英文ok的同学可以直接参考。以下会举一个例子说明如何使用gensim，可以作为官方例子的补充：
```
$ python
>>> from gensim import corpora, models, similarities
>>> import logging
>>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
>>> documents = ["Shipment of gold damaged in a fire", "Delivery of silver arrived in a silver truck", "Shipment of gold arrived in a truck"]
```

正常情况下，需要对文本做一些预处理，譬如去停用词，对文本进行tokenize，stemming以及过滤掉低频的词，但是为了说明问题，以下的预处理仅仅是将英文单词小写化：
```
>>> texts = [[word for word in document.lower().split()] for document in documents]
>>> print texts
[['shipment', 'of', 'gold', 'damaged', 'in', 'a', 'fire'], ['delivery', 'of', 'silver', 'arrived', 'in', 'a', 'silver', 'truck'], ['shipment', 'of', 'gold', 'arrived', 'in', 'a', 'truck']]
```

我们可以通过这些文档抽取一个“词袋:bag-of-words”，将文档的token映射为id：
```
>>> dictionary = corpora.Dictionary(texts)
>>> print dictionary
Dictionary(11 unique tokens: [u'a', u'damaged', u'gold', u'fire', u'of']...)
>>> print dictionary.token2id
{u'a': 0, u'damaged': 1, u'gold': 2, u'fire': 3, u'of': 4, u'delivery': 7, u'truck': 8, u'shipment': 5, u'in': 6, u'arrived': 9, u'silver': 10}
```

然后就可以将用字符串表示的文档转换为用id表示的文档向量：
```
>>> corpus = [dictionary.doc2bow(text) for text in texts]
>>> print corpus
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)], [(0, 1), (4, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 2)], [(0, 1), (2, 1), (4, 1), (5, 1), (6, 1), (8, 1), (9, 1)]]
```

例如`(10,2)`这个元素代表第二篇文档中id为10的单词“silver”出现了2次。有了这些信息，我们就可以基于这些“训练文档”计算一个TF-IDF“模型”：
```
>>> tfidf = models.TfidfModel(corpus)
```

基于这个TF-IDF模型，我们可以将上述用词频表示文档向量表示为一个用tf-idf值表示的文档向量：
```
>>> corpus_tfidf = tfidf[corpus]
>>> for doc in corpus_tfidf:
...     print doc
... 
[(1, 0.6633689723434505), (2, 0.2448297500958463), (3, 0.6633689723434505), (5, 0.2448297500958463)]
[(7, 0.4355066251613605), (8, 0.16073253746956623), (9, 0.16073253746956623), (10, 0.871013250322721)]
[(2, 0.5), (5, 0.5), (8, 0.5), (9, 0.5)]
```

发现一些token貌似丢失了，我们打印一下tfidf模型中的信息：
```
>>> print tfidf.dfs
{0: 3, 1: 1, 2: 2, 3: 1, 4: 3, 5: 2, 6: 3, 7: 1, 8: 2, 9: 2, 10: 1}
>>> print tfidf.idfs
{0: 0.0, 1: 1.5849625007211563, 2: 0.5849625007211562, 3: 1.5849625007211563, 4: 0.0, 5: 0.5849625007211562, 6: 0.0, 7: 1.5849625007211563, 8: 0.5849625007211562, 9: 0.5849625007211562, 10: 1.5849625007211563}
```

我们发现由于包含id为0，4，6这3个单词的文档数(df)为3，而文档总数也为3，所以idf被计算为0了，看来gensim没有对分子加1，做一个平滑。不过我们同时也发现这3个单词分别为a, in, of这样的介词，完全可以在预处理时作为停用词干掉，这也从另一个方面说明TF-IDF的有效性。

有了tf-idf值表示的文档向量，我们就可以训练一个LSI模型，我们设置topic数为2：
```
>>> lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
>>> lsi.print_topics(2)
[(0, u'-0.438*"gold" + -0.438*"shipment" + -0.366*"truck" + -0.366*"arrived" + -0.345*"damaged" + -0.345*"fire" + -0.297*"silver" + -0.149*"delivery" + 0.000*"a" + 0.000*"in"'), 
(1, u'-0.728*"silver" + 0.364*"damaged" + 0.364*"fire" + -0.364*"delivery" + 0.134*"shipment" + 0.134*"gold" + -0.134*"truck" + -0.134*"arrived" + -0.000*"a" + 0.000*"in"')]
```

lsi的物理意义不太好解释，不过最核心的意义是将训练文档向量组成的矩阵SVD分解，并做了一个秩为2的近似SVD分解。有了这个lsi模型，我们就可以将文档映射到一个二维的topic空间中：
```
>>> corpus_lsi = lsi[corpus_tfidf]
>>> for doc in corpus_lsi:
...     print doc
... 
[(0, -0.67211468809878527), (1, 0.5488068211935605)]
[(0, -0.44124825208697882), (1, -0.83594920480339008)]
[(0, -0.80401378963792725)]
```

可以看出，文档1，3和topic1更相关，文档2和topic2更相关。我们也可以顺手跑一个LDA模型：
```
>>> lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
>>> lda.print_topics(2)
[(0, u'0.134*silver + 0.112*arrived + 0.112*truck + 0.110*shipment + 0.107*gold + 0.095*delivery + 0.085*fire + 0.075*damaged + 0.056*a + 0.056*of'), 
(1, u'0.132*damaged + 0.120*fire + 0.105*gold + 0.102*shipment + 0.090*silver + 0.088*truck + 0.088*arrived + 0.078*delivery + 0.066*a + 0.066*of')]
```

lda模型中的每个主题单词都有概率意义，其加和为1，值越大权重越大，物理意义比较明确，不过反过来再看这三篇文档训练的2个主题的LDA模型太平均了，没有说服力。

好了，我们回到LSI模型，有了LSI模型，我们如何来计算文档直接的相似度，或者换个角度，给定一个查询Query，如何找到最相关的文档？当然首先是建索引了：
```
>>> index = similarities.MatrixSimilarity(corpus_lsi)
```

以查询Query为例：gold silver truck。首先将其向量化：
```
>>> query = "gold silver truck"
>>> query_bow = dictionary.doc2bow(query.lower().split())
>>> print query_bow
[(2, 1), (8, 1), (10, 1)]
```

再用之前训练好的LSI模型将其映射到二维的topic空间：
```
>>> query_tfidf = tfidf[query_bow]
>>> query_lsi = lsi[query_tfidf]
>>> print query_lsi
[(0, -0.5265936353666032), (1, -0.64548839066948904)]
```

最后就是计算其和index中doc的余弦相似度了：
```
>>> sims = index[query_lsi]
>>> print list(enumerate(sims))
[(0, -0.00043871999), (1, 0.98033714), (2, 0.63213468)]
```

当然，我们也可以按相似度进行排序：
```
>>> sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
>>> print sort_sims
[(1, 0.98033714), (2, 0.63213468), (0, -0.00043871999)]
```

## 课程图谱相关实验
前面用一个简单的例子过了一遍gensim的用法，接下来我们将用课程图谱的实际数据来做一些验证和改进，同时会用到NLTK来对课程的英文数据做预处理。

### 数据准备
感谢原作者的分享的一份Coursera的课程数据，[百度网盘链接](http://t.cn/RhjgPkv)，提取码`oppc`。总共379个课程，每行包括3部分内容：课程名，课程简介，课程详情。

首先让我们打开Python，加载这份数据：
```
>>> import os
>>> os.getcwd()
>>> os.chdir('/home/flystar/lab-wiki')
>>> courses = [line.strip() for line in file('coursera_corpus')]
>>> courses_name = [course.split('\t')[0] for course in courses]
>>> print courses_name[0:10]
```

### 引入NLTK
NTLK是著名的Python自然语言处理工具包，但是主要针对的是英文处理。NLTK配套有文档，有语料库，有书籍，甚至国内有同学无私的翻译了这本书：“用Python进行自然语言处理”，有时候不得不感慨：做英文自然语言处理的同学真幸福。

首先仍然是安装NLTK。可以import测试一下，如果没有问题，还有一件非常重要的工作要做，下载NLTK官方提供的相关语料：
```
$ sudo pip install -U nltk
>>> import nltk
>>> nltk.download()
```

这个时候会弹出一个图形界面，会显示两份数据供你下载，分别是all-corpora和book，最好都选定下载了，这个过程需要一段时间，语料下载完毕后，NLTK在你的电脑上才真正达到可用的状态，可以测试一下布朗语料库：
```
>>> from nltk.corpus import brown
>>> brown.readme()
>>> brown.words()[0:10]
>>> brown.tagged_words()[0:10]
>>> len(brown.words())
```

现在我们就来处理刚才的课程数据，如果按此前的方法仅仅对文档的单词小写化的话，我们将得到如下的结果：
```
>>> texts_lower = [[word for word in document.lower().split()] for document in courses]
>>> print texts_lower[0]
```

注意其中很多标点符号和单词是没有分离的，所以我们引入nltk的`word_tokenize`函数，并处理相应的数据：
```
>>> from nltk.tokenize import word_tokenize
>>> texts_tokenized = [[word.lower() for word in word_tokenize(document.decode('utf-8'))] for document in courses]
>>> print texts_tokenized[0]
```

对课程的英文数据进行tokenize之后，我们需要去停用词，幸好NLTK提供了一份英文停用词数据：
```
>>> from nltk.corpus import stopwords
>>> english_stopwords = stopwords.words('english')
>>> print english_stopwords
>>> len(english_stopwords)
```

总计153个停用词，我们首先过滤课程语料中的停用词：
```
>>> texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
>>> print texts_filtered_stopwords[0]
```

停用词被过滤了，不过发现标点符号还在。我们首先定义一个标点符号list，然后过滤这些标点符号：
```
>>> english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
>>> texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]
>>> print texts_filtered[0]
```

更进一步，我们对这些英文单词词干化(Stemming)，NLTK提供了好几个相关工具接口可供选择，可选的工具包括Lancaster Stemmer、Porter Stemmer等知名的英文Stemmer。这里我们使用Lancaster Stemmer：
```
>>> from nltk.stem.lancaster import LancasterStemmer
>>> st = LancasterStemmer()
>>> st.stem('stemmed')
>>> st.stem('stemming')
>>> st.stem('stemmer')
```

让我们调用这个接口来处理上面的课程数据：
```
>>> texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]
>>> print texts_stemmed[0]
```

在引入gensim之前，还有一件事要做，去掉在整个语料库中出现次数为1的低频词，测试了一下，不去掉的话对效果有些影响(提供了两种方案生成texts，选其一即可)：
```
>>> from collections import defaultdict
>>> frequency = defaultdict(int)
>>> for text in texts_stemmed:
...     for token in text:
...         frequency[token] += 1
... 
>>> texts = [[token for token in text if frequency[token] > 1] for text in texts_stemmed]
>>> from pprint import pprint
>>> pprint(texts)
>>> all_stems = sum(texts_stemmed, [])
>>> stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
>>> texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
```

### 引入gensim
有了上述的预处理，我们就可以引入gensim，并快速的做课程相似度的实验了。以下会快速的过一遍流程，具体的可以参考上一节的详细描述。
```
>>> from gensim import corpora, models, similarities
>>> import logging
>>> logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
>>> dictionary = corpora.Dictionary(texts)
>>> corpus = [dictionary.doc2bow(text) for text in texts]
>>> tfidf = models.TfidfModel(corpus)
>>> corpus_tfidf = tfidf[corpus]
>>> lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
>>> index = similarities.MatrixSimilarity(lsi[corpus_tfidf])
```

基于LSI模型的课程索引建立完毕，我们以Andrew Ng教授的机器学习公开课为例，这门课程在我们的coursera_corpus文件的第211行，现在我们就可以通过lsi模型将这门课程映射到10个topic主题模型空间上，然后和其他课程计算相似度：
```
>>> print courses_name[210]
>>> ml_course = texts[210]
>>> ml_bow = dicionary.doc2bow(ml_course)
>>> ml_tfidf = tfidf[ml_bow]
>>> ml_lsi = lsi[ml_tfidf]
>>> print ml_lsi
>>> sims = index[ml_lsi]
>>> print list(enumerate(sims))
>>> sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
>>> print sort_sims[0:10]
>>> print courses_name[210] #它自己
>>> print courses_name[174] #大牛Pedro Domingos机器学习公开课
>>> print courses_name[238] #大牛Daphne Koller教授的概率图模型公开
>>> print courses_name[203] #大牛Geoffrey Hinton的神经网络公开课
```

此文非原创，感谢原作者无私分享。[原文链接](http://www.52nlp.cn/如何计算两个文档的相似度三)