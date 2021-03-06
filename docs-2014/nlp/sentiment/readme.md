title: 基于词典的中文情感分析
date: 2015-05-18
tags: [NLP,情感分析]
---
目前，情感倾向分析的方法主要分为两类：一种是基于情感词典的方法；一种是基于机器学习的方法。前者需要用到标注好的情感词典。后者则需要大量的人工标注的语料作为训练集，通过提取文本特征，构建分类器来实现情感的分类。

<!--more-->
## 算法设计思路
**1 文本切割**
首先，将文档以换行符分割成不同段落；其次，将段落用中文里最常用的句号、分号、问号、感叹号等划分句意的符号，切割成不同句子；再次，用逗号划分出句子里的意群(表示情感的最小单元)。

**2 中文分词**
对意群进行分词。开源中文分词工具有很多，如在线的SCWS(PHP)，张华平博士团队开发的NLPIR(C、Python、Java)，哈工大的LTP(C++、Python)，还有R语言的分词包RWordseg。几款分词工具各有各自的特点，比如SCWS分词可以获得每个词的IDF值。

**3 情感分析**
- **情感词**。情感分析是从发现句子中的情感词开始。`操作：将分词后得到的词依次在情感词典中查找，若能找到，则读取情感极性及权值(s=性感极性x权值)，否则不是情感词。`
- **程度词**。当程度词修饰情感词，文本的情感倾向度会发生变化。为了准确表达文本的情感，需做相应的调整。`操作：情感词往前搜索程度词，找到程度词或遇到情感词就停止搜寻，若找到程度词则记录相应权值(k)。`
- **否定词**。否定词的处理分两种情况，一是修饰情感词，一是修饰程度词。

比如：
```
"我很不高兴" 分词：我 很 不 高兴
"我不很高兴" 分词：我 不 很 高兴
```

可以看出，第一句话表达的是一种很强烈的负面情感，第二句话表达的是一种较弱的正面情感。因此，如果否定词在程度词之前，起到的是减弱的作用；如果否定词在程度词之后，则起到的是逆向情感的作用。`操作：情感词往前搜索否定词，搜索完当前意群或遇到情感词就停止。逐个处理否定词，若位于情感词前，则取wi=-1，若位于程度词前，则取wi=0.5。`

**4 情感聚合**
通过前面的操作，已经完成意群划分，同时也提出了情感词、程度词和否定词，并赋予了相应的权值。有了这些，就可以计算 **意群情感值**` = sum(情感词1,情感词2,..); 情感词1 = s x k x ( w1 x .. x wn)`。

**5 输出结论**
文档的情感值一般是累加各个意群情感值。但很多时候文档里有褒有贬，若不想失去这些重要信息，可以输出一个积极情感分值和一个消极情感分值。如果感兴趣还可以输出积极情感均值，消极情感均值，积极情感方差，消极情感方差等。
>上述的做法是最简单的做法，并没有考虑太多句子之间的差异以及不同段落对文档的重要性等情况。

## 后话情感词典
情感词典包括基础情感词典、拓展情感词典和领域情感词典。

**1 基础情感词典**
基础情感词典包括了一些被广泛认同的情感词，比如好、漂亮、差、烂这些词。有研究者已经帮我们整理了这么一份情感词典。一个是著名的知网(Hownet)情感词典，还有一个是台湾大学简体中文情感极性词典(NTUSD)。
把知网(Hownet)里面的正面评价词语、正面情感词语和NTUSD的积极词典消重之后组合在一起，成为基础积极情感词典。
把知网(Hownet)里面的负面评价词语、负面情感词语和NTUSD的消极词典消重之后组合在一起，成为基础消极情感词典。
另外，需要对知网(Hownet)里面的程度级别词语进行权值的设置。
>停用词表一般使用哈工大的停用词表，网上有下载的资源。

**2 拓展情感词典**
拓展情感词典其实就是把基础情感词典通过同义词词典找到情感词的同义词，这样就拓展了基础情感词典。

**3 领域情感词典**
仅仅依靠基础情感词典来识别一个句子里面的情感词是不足够的。在特定的领域，有些并非基础的情感词也有情绪倾向。比如：“这手机很耐摔啊，还防水”。“耐摔”、“防水”就是在手机这个领域有积极情绪的词。

**要怎么识别这些词呢？一般使用的方法是PMI(互信息)方法。**简单的说，如果一个词和积极的词语共现的频率高，那么这个词是积极倾向的可能性也会大，反之亦然。所以，只要计算一个词和积极词共现的频率和消极词共现的频率之差，并设定某个阈值，就可以粗略的得知这个词的情感倾向了。

计算共现又可以细分两种方法：一种是利用搜索引擎计算共现值，一种是直接利用语料计算共现值。具体方法：

1. 先选定核心情感词(可以有多个)，该核心情感词的情感必须非常明确，具有代表性。如：“好”、“烂”。
2. 利用搜索引擎计算共现值。即在搜索引擎中搜索“某个词+好”，记录下网页数量co_pos。然后再搜索“某个词+烂”，记录下网页数量co_neg。再搜索“某个词”，记录下网页数量n。再搜索“好”，网页数量为pos，搜索“烂”，网页数量为neg。由此可利用这些数据来计算积极互信息和消极互信息。最后求两个互信息之差，差为正则积极、为负则消极。重复计算不同词的互信息之差，最后选分值高的即可组成领域情感词典。
3. 利用语料库计算共现值。原理一样，在语料库中搜索“某个词+好”，记录下数量。再搜索“某个词+烂”，记录下数量。后面的步骤都和上面一样。

**4 最后把三个词典结合起来，就形成了完整的情感词典。**
最后必须说明，利用情感词典来判断一个句子的情感是有着明显不足的。中文有着丰富的语义表达，很多情感都是隐含的，比如：“我昨天吃了这道菜，今天就拉肚子了”。这句话没有一个情感词，但表达的是消极的情绪。还有各种事正话反说的句子，比如：“你说这里的菜很好吃，我只能呵呵了”。如果用词典匹配，有“好吃”、“呵呵”两个积极词，但这句话表达的绝非积极的情绪。这时就需要更高级复杂的处理方式，要更深入句子的句法，语法了。

## 参考资料：
- [基于词典的中文情感倾向分析算法设计](http://site.douban.com/146782/widget/notes/15462869/note/355625387/)
- [Python文本挖掘：使用情感词典进行情感分析·算法设计](http://rzcoding.blog.163.com/blog/static/2222810172013101844033170/)
- [Python文本挖掘：使用情感词典进行情感分析·情感词典](http://rzcoding.blog.163.com/blog/static/2222810172013101991918346/)
- [电商信息化技巧：基于用户评论的情感分析算法](http://www.itseo.net/direction/show-114.html)