title: N最短路径分词
date: 2016-07-14
tags: [NLP,NShort,分词]
---
中文分词一般包括3个过程：预处理过程的词语粗切分，切分排歧与未登录词识别，词性标注。面对歧义、人名、地名、译名、实体识别等多种计算机难于处理的语言现象，一个稳妥的办法便是将所有可能的分词结果一并给出，让后续的程序来进一步处理这些结果。此预处理过程便是本文讨论的重点。

<!--more-->
## N最短路径的粗分模型
汉语很美，她总是力图使用最少的字来表达最丰富的含义。所以，一个中文句子总是词越少越好，这就是这个算法寻求最短路径的依据所在。

**1.1 基本思想**
根据词典，找出字串中所有可能的词，构造词语切分有向无环图。每个词对应图中的一条有向边，并赋给相应的边长(权值)。然后针对该切分图，在起点到终点的所有路径中，找出N最短路径作为粗分结果集。如果两条或两条以上路径长度相等，那么他们的长度并列第i，都要列入粗分结果集，而且不影响其他路径的排列序号，最后的粗分结果集合大小大于或等于N。

**1.2 模型求解**
设待分字串$S=c_{1}c_{2}..c_{n}$，其中$c_i$为单个字，n为串的长度，建立节点数为`n+1`的有向无环图G，各节点依次为$V_{0},V_{1},..,V_{n}$。通过以下两种方法建立G所有可能的词边：

1. 相邻节点$V_{k-1},V_{k}$之间建立有向边$<V_{k-1},V_{k}>$，边的长度为$L_k$，边对应默认词$c_i$；
2. 若$w=c_{i}c_{i+1}..c_{j}$是一个词，则节点$V_{i-1},V_{j}$之间建立有向边$<V_{i-1},V_{j}>$，边的长度为$L_w$，边对应的词为w。`(0<i<j<=n)`

这样，待分字串S中包含的所有词与有向无环图G中的边对应。在粗分模型中，我们假定所有的词都是对等的，为了计算方便，将词的对应边长的边长均设为1。

## N最短路径的统计粗分模型
在非统计模型构建粗切有向无环图有向边的过程中，我们给每个词的对应边长度赋值为1。随着字串长度n和最短路径数N的增大，长度相同的路径数急剧增加，同时粗切分结果数量必然上升。大量的切分结果对后期的处理，以及整个性能的提高是非常不利的。

**2.1 基本原理**
假定一个词串W经过信道传送，由于噪声干扰而丢失了词界的切分标志，到输出端便成了汉字串C，这是一个典型的噪声信道问题。N最短路径方法词语粗分模型可以相应的改进为：求取W，使得概率P(W|C)的值是最大N个。为了简化P(W|C)的计算，我们采用的是一元统计模型，即只引入词频并假定词与词之间是相互独立。基于以上分析，我们引入词$w_i$的词频信息$P(w_i)$，对模型进行了改进，得到一个基于N最短路径的一元统计模型。

**2.2 模型求解**

\begin{align}
P(W|C) = P(W)P(C|W)/P(C)
\end{align}

其中，$P(C)$是汉字串的概率，它是一个常数，不必考虑。从词串恢复到汉字串的概率$P(C|W)=1$，因为只有唯一的方式。因此目标就是确定$P(W)$最大N种的切分结果集合。

$W=w_{1}w_{2}..w_{m}$是字串$S=c_{1}c_{2}..c_{n}$的一种粗分结果。$w_i$是一个词，$P(w_i)$表示其出现的概率。在大规模语料库训练的基础上，根据大数定理有：

\begin{align}
P(w_i) \approx k_i / \sum_{j=1}^{m} k_j
\end{align}

其中$k_i$为$w_i$在训练样本中出现的次数。粗切分阶段，为了简单处理，我们仅仅采取了概率上下文无关文法，即假设上下文无关，词与词之间相互独立。因此，我们可以得到：

\begin{align}
P(W) = \prod_{i=1}^{m} P(w_i) \approx \prod_{i=1}^{m} k_i / \sum_{j=1}^{m} k_j
\end{align}

为处理方便，令：

\begin{align}
NP(W) = - \ln P(W) = \sum_{i=1}^{m} - \ln P(w_i)
\end{align}

那么也就将$P(W)$极大值的问题转化为$NP(W)$极小值的问题。适当修改有向无环图G边的长度(加1，数据简单平滑)：

1. $<V_{k-1},V_{k}>$的长度值$L_{k}=- \ln (0+1)$；
2. $w=c_{i}c_{i+1}..c_{j}$对应有向边$<V_{i-1},V_{j}>$，其长度值$L_{w}= \ln (\sum_{j=1}^{m} k_{j} +m) - \ln (k_{i}+1)$。

针对修改了边长后的有向无环图G，就可以实现问题的最终求解，求出从起点到终点排名前N的最短路径。

## N最短路径
Dijkstra算法是典型的单源最短路径算法，用于计算一个节点到其他所有节点的最短路径。设`G=(V,E)`是一个带权有向图，把图中顶点集合V分成S、U两组，算法步骤：

1. 初始时，S只包含源点，即`S＝{v}`，v的距离为0。U包含除v外的其他顶点，若v与U中顶点u有边，则`<u,v>`正常有权值，若u不是v的出边邻接点，则`<u,v>`权值为无穷大；
2. 从U中选取一个距离v最小的顶点k，把k加入S中，同时从U中移除顶点k；
3. 以k为新考虑的中间点，修改U中各顶点的距离。若从源点v到顶点u的距离(经过顶点k)比原来距离(不经过顶点k)短，则修改顶点u的距离值；
4. 重复步骤`2`和`3`，直到所有顶点都包含在S中。

计算前N最短路径和Dijkstra主要差别是每个顶点要记录多个前驱，和对应的权重，在Dijkstra里面只是记录一个。

## 参考资料：
- [基于N-最短路径方法的中文词语粗分模型，张华平、刘群](http://xueshu.baidu.com/)
- [最短路径-Dijkstra算法和Floyd算法](http://www.cnblogs.com/biyeymyhjob/archive/2012/07/31/2615833.html)
- [N最短路径的Java实现与分词应用](http://www.hankcs.com/nlp/segment/n-shortest-path-to-the-java-implementation-and-application-segmentation.html)