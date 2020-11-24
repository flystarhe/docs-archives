title: Ngram语言模型
date: 2016-08-16
tags: [NLP,Ngram,语言识别]
---
Ngram(也称为N元模型)是自然语言处理中一个非常重要的概念。在NLP中，人们基于一定的语料库，可以利用Ngram来预计或者评估一个句子是否合理。另外一方面，Ngram可以用来评估两个字符串之间的差异程度，这是模糊匹配中常用的一种手段。而且广泛应用于机器翻译、语音识别、印刷体和手写体识别、拼写纠错、汉字输入和文献查询。

<!--more-->
## Ngram Model
假定S表示某个有意义的句子，由一串特定顺序排列的词w1,w2,w3,..,wn组成，n是句子的长度。想知道S在文本中(语料库)出现的可能性，也就是数学上所说的概率P(S)：

\begin{align}
P(S) &= P(w1,w2,w3,..,wn) \\
     &= P(W1)P(W2|W1)P(W3|W1,W2)..P(Wn|W1,W2,..,Wn-1)
\end{align}

可是这样的方法存在两个致命的缺陷：

1. 參数空间过大：条件概率P(wn|w1,w2,..,wn-1)的可能性太多，无法估算，不可能有用；
2. 数据稀疏严重：对于非常多词对的组合，在语料库中都没有出现，依据最大似然估计得到的概率将会是0。最后的结果是，我们的模型仅仅能算可怜兮兮的几个句子，而大部分的句子算得的概率是0。

## 马尔科夫假设
为了解决參数空间过大的问题。引入了马尔科夫假设：随意一个词出现的概率只与它前面出现的有限的一个或者几个词有关。

如果一个词的出现仅依赖于它前面出现的一个词，那么我们就称之为bigram：

\begin{align}
P(S) &= P(w1,w2,w3,..,wn) \\
     &= P(W1)P(W2|W1)P(W3|W1,W2)..P(Wn|W1,W2,..,Wn-1) \\
     &\approx P(W1)P(W2|W1)P(W3|W2)..P(Wn|Wn-1)
\end{align}

假设一个词的出现仅依赖于它前面出现的两个词，那么我们就称之为trigram：

\begin{align}
P(S) &= P(w1,w2,w3,..,wn) \\
     &= P(W1)P(W2|W1)P(W3|W1,W2)..P(Wn|W1,W2,..,Wn-1) \\
     &\approx P(W1)P(W2|W1)P(W3|W2,W1)..P(Wn|Wn-1,Wn-2)
\end{align}

一般来说，N元模型就是假设当前词的出现概率只与它前面的N-1个词有关。而这些概率参数都是可以通过大规模语料库来计算，比如三元概率有：

\begin{align}
P(W_i|W_{i-1},W_{i-2}) \approx count(W_{i-2}W_{i-1}W_i) / count(W_{i-2}W_{i-1})
\end{align}

在实践中用的最多的就是bigram和trigram了，高于四元的用的非常少，由于训练它须要更庞大的语料，并且数据稀疏严重，时间复杂度高，精度却提高的不多。

## 数据平滑
有研究人员用150万词的训练语料来训练trigram模型，然后用同样来源的测试语料来做验证，结果发现23%的trigram没有在训练语料中出现过。对语言而言，由于数据稀疏的存在，极大似然法不是一种很好的参数估计办法。这时的解决办法，我们称之为“平滑技术”。(參考《数学之美》第33页)

数据平滑的目的有两个：一个是使全部的Ngram概率之和为1；一个是使全部的Ngram概率都不为0。其主要策略是把在训练样本中出现过的事件的概率适当减小，然后把减小得到的概率密度分配给训练语料中没有出现过的事件。实际中平滑算法有很多种，例如：

- Add-one平滑
- Witten-Bell平滑
- Good-Turing平滑
- Katz Backoff
- Stupid Backoff

数据平滑技术是构造高鲁棒性语言模型的重要手段，且数据平滑的效果与训练语料库的规模有关。训练语料库规模越小，数据平滑的效果越显著；训练语料库规模越大，数据平滑的效果越不显著，甚至可以忽略。

## 基于Ngram模型定义的字符串距离
模糊匹配的关键在于如何衡量两个长得很像的单词(或字符串)之间的“差异”，这种差异通常又称为“距离”。除了可以定义两个字符串之间的编辑距离(通常利用Needleman-Wunsch算法或Smith-Waterman算法)，还可以定义它们之间的Ngram距离。

假设有一个字符串S，那么该字符串的Ngram就表示按长度N切分原词得到的词段，也就是S中所有长度为N的子字符串。设想如果有两个字符串，然后分别求它们的Ngram，那么就可以从它们的共有子串的数量这个角度去定义两个字符串间的Ngram距离。但是仅仅是简单地对共有子串进行计数显然也存在不足，这种方案显然忽略了两个字符串长度差异可能导致的问题。比如字符串girl和girlfriend，二者所拥有的公共子串数量显然与girl和其自身所拥有的公共子串数量相等，但是我们并不能据此认为girl和girlfriend是两个等同的匹配。为了解决该问题，有学者提出以非重复的Ngram分词为基础来定义Ngram距离，公式表示如下：

\begin{align}
|G_N(S_1)| + |G_N(S_2)| - 2 \times |G_N(S_1) \cap G_N(S_2)|
\end{align}

此处，$|G_N(S_1)|$是字符串$S_1$的Ngram集合，N值一般取2或者3。以N=2为例对字符串Gorbachev和Gorbechyov进行分段，可得如下结果：
```
Go or rb ba ac ch he ev
Go or rb be ec ch hy yo ov
```

结合上面的公式，即可算得两个字符串之间的距离是`8 + 9 − 2 × 4 = 9`。显然，字符串之间的距离越小，它们就越接近。当两个字符串完全相等的时候，它们之间的距离就是0。

## 利用Ngram模型评估语句是否合理
从统计的角度来看，自然语言中的一个句子S可以由任何词串构成，不过概率P(S)有大有小。例如：

- S1 = 我刚吃过晚饭
- S2 = 刚我过晚饭吃

显然，对于中文而言S1是一个通顺而有意义的句子，而S2则不是，所以对于中文来说P(S1)>P(S2)。

另外一个例子是，如果我们给出了某个句子的一个节选，我们其实可以能够猜测后续的词应该是什么，例如：

- the large green __ .  mountain or tree ?
- Kate swallowed the large green __ .  pill or broccoli ?

假设我们现在有一个语料库如下，其中`<s1><s2>`是句首标记，`</s2></s1>`是句尾标记：
```
<s1><s2>yes no no no no yes</s2></s1>
<s1><s2>no no no yes yes yes no</s2></s1>
```

下面我们的任务是来评估如下这个句子的概率：
```
<s1><s2>yes no no yes</s2></s1>
```

我们来演示利用trigram模型来计算概率的结果：

\begin{align}
&P(yes|<s1>,<s2>) = 1/2, &P(no|<s2>,yes) = 1 \\
&P(no|yes,no) = 1/2,     &P(yes|no,no) = 2/5 \\
&P(</s2>|no,yes) = 1/2,  &P(</s1>|yes,</s2>) = 1
\end{align}

所以我们要求的概率就等于：

\begin{align}
1/2 \times 1 \times 1/2 \times 2/5 \times 1/2 \times 1 = 0.05
\end{align}

## 基于Ngram模型的文本分类器
Ngram如何用作文本分类器的呢？其实很简单了，只要根据每个类别的语料库训练各自的语言模型，实质上就是每一个类别都有一个概率分布，当新来一个文本的时候，只要根据各自的语言模型，计算出每个语言模型下这篇文本的发生概率，文本在哪个模型的概率大，这篇文本就属于哪个类别了！

## Ngram在语言识别中的应用
Ngram是从文本或文档中提取的字符或单词序列，可被分成两组：基于字符的或基于单词的。一个Ngram是提取自一个单词（在我们例子中是一个字符串）的一组N个连续字符。其后的动机是类似的单词将具有高比例的Ngram。最常见的N值是2和3，分别称为bigram和trigram。比如，单词`TIKA`生成的bigram为`*T、TI、IK、KA、A*`，生成的trigram为`**T、*TI、TIK、IKA、KA*、A**`。`*`代表的是一个补充空间。基于字符的Ngram被用来量度字符串的相似性。使用基于字符的Ngram有些应用程序有拼写检查程序、stemming和OCR。

单词Ngram是提取自文本的N个连续单词的序列。它也独立于语言。基于两个字符串间的相似性的Ngram是由Dice的系数衡量的。`s = (2|X /\ Y|)/(|X| + |Y|)`，其中X和Y是这个字符集。`/\`表示两个集间的一个交集。

**语言的识别是怎样实现的？**总的来说，当要判断一个新的文档是用的什么语言时，我们首先要创建文档的Ngram概要文件并算出这个新文档概要文件与语言概要文件之间的距离。这个距离的计算根据的是两个概要文件之间的“out-of-place measure”。选择最短的距离，它表示此特定的文档属于该语言。这里要引入一个阈值，它的作用是当出现任何超过阈值的距离时，系统就会报告这个文档的语言不能被判定或判定有误。

[Apache/Tika](http://tika.apache.org/)能够帮助识别一段文字的语言，在元数据不包括语言信息时非常有用。这里使用[optimaize/language-detector](https://github.com/optimaize/language-detector)演示：
```
import java.util
import com.optimaize.langdetect.{DetectedLanguage, LanguageDetectorBuilder}
import com.optimaize.langdetect.ngram.NgramExtractors
import com.optimaize.langdetect.profiles.LanguageProfileReader

object Test {
  def main(args: Array[String]) {
    //case one: load all languages
    val languageProfiles1 = new LanguageProfileReader().readAllBuiltIn()
    val languageDetector1 = LanguageDetectorBuilder.create(NgramExtractors.standard()).withProfiles(languageProfiles1).build()
    val rs1 = languageDetector1.getProbabilities("正如您猜想的那样，单词 N-gram 是提取自文本的 N 个连续单词的序列。它也独立于语言。")
    show(rs1)
    //case two: load "en,zh-CN" language
    val langs = new util.ArrayList[String]()
    langs.add("en");langs.add("zh-CN");
    val languageProfiles2 = new LanguageProfileReader().read(langs)
    val languageDetector2 = LanguageDetectorBuilder.create(NgramExtractors.standard()).withProfiles(languageProfiles2).build()
    val rs2 = languageDetector2.getProbabilities("正如您猜想的那样，单词 N-gram 是提取自文本的 N 个连续单词的序列。它也独立于语言。")
    show(rs2)
  }

  def show(list: java.util.List[DetectedLanguage]): Unit = {
    if (list.isEmpty) {
      println("err", "none")
    } else {
      val best = list.get(0)
      println(best.getLocale.getLanguage, best.getProbability)
    }
  }
}
```

建议输入字符数在50以上，概要文件内容过少会影响识别效果。具体语言名称请查询[编码名称语言对照表](http://www.loc.gov/standards/iso639-2/php/code_list.php)。

## ARPA格式的Ngram语言模型
先看一下语言模型的格式：(注：值都是以10为底的对数值)
```
\data\  
ngram 1=64000  
ngram 2=522530  
ngram 3=173445  

\1-grams:  
-5.24036        'cause  -0.2084827  
-4.675221       'em     -0.221857  
-4.989297       'n      -0.05809768  
-5.365303       'til    -0.1855581  
```

上面是一个语言模型的一部分，三元语言模型的综合格式如下：
```
\data  
ngram 1=nr # 一元语言模型  
ngram 2=nr # 二元语言模型  
ngram 3=nr # 三元语言模型  

\1-grams:  
pro_1 word1 back_pro1  

\2-grams:  
pro_2 word1 word2 back_pro2  

\3-grams:  
pro_3 word1 word2 word3  

\end\  
```

首项是Ngram的条件概率，即P(wordN|word1,word2,..,wordN-1)，末项是回退的权重，中间为Ngram的词。

举例来说，对于三个连续的词来说，P(word3|word1,word2)：
```
if(存在(word1,word2,word3)的三元模型){
    return pro_3(word1,word2,word3);
}else if(存在(word1,word2)的二元模型){
    return back_pro2(word1,word2)*P(word3|word2);
}else{
    return P(word3|word2);
}
```

对于二个连续的词来说，P(word2|word1)：
```
if(存在(word1,word2)的二元模型){
    return pro_2(word1,word2);
}else{
    return back_pro2(word1)*pro_1(word2);
}
```

## Ngram语言模型训练工具SRILM
SRILM的主要目标是支持语言模型的估计和评测。估计是从训练数据中得到一个模型，包括最大似然估计及相应的平滑算法；而评测则是从测试集中计算其困惑度。其最基础和最核心的模块是Ngram模块，包括两个工具：ngram-count和ngram，相应的被用来估计语言模型和计算语言模型的困惑度。一个标准的语言模型(三元语言模型，使用Good-Truing打折法和katz回退进行平衡)可以用如下的命令构建：
```
$ ngram-count -text TRAINDATA -lm LM
其中LM是输出的语言模型文件，可以用如下的命令进行评测：
$ ngram -lm LM -ppl TESTDATA -debug 2
```

从语料库中生成n-gram计数文件：
```
$ ngram-count -text europarl.en -order 3 -write europarl.en.count
```

其中参数-text指向输入文件，此处为europarl.en；-order指向生成几元的Ngram，即N=3；-write指向输出文件，此处为europarl.en.count。

从上一步生成的计数文件中训练语言模型：
```
$ ngram-count -read europarl.en.count -order 3 -lm europarl.en.lm -interpolate -kndiscount
```

其中参数-read指向输入文件，此处为europarl.en.count；-order与上同；-lm指向训练好的语言模型输出文件，此处为europarl.en.lm；最后两个参数为所采用的平滑方法，-interpolate为插值平滑，-kndiscount为`modified Kneser-Ney`打折法，这两个是联合使用的。一般我们训练语言模型时，这两步是合二为一的，这里主要是为了介绍清楚Ngram语言模型训练的步骤细节。语言模型europarl.en.lm为ARPA文件格式。

利用上一步生成的语言模型计算测试集的困惑度：
```
$ ngram -ppl devtest2006.en -order 3 -lm europarl.en.lm > europarl.en.lm.ppl
```

其中测试集采用devtest2006.en，2000句；参数-ppl为对测试集句子进行评分（logP(T)，其中P(T)为所有句子的概率乘积）和计算测试集困惑度的参数；europarl.en.lm.ppl为输出结果文件；其他参数同上。

对于大文本的语言模型训练不能使用上面的方法，主要思想是将文本切分，分别计算，然后合并。步骤如下：
```
$ split -l 10000 trainfile.txt filedir/
#每10000行数据为一个新文本存到filedir目录
$ make-bath-counts filepath.txt 1 cat ./counts -order 3
#将统计的词频结果存放在counts目录。其中filepath.txt为切分文件的全路径
$ merge-batch-counts ./counts
#合并counts文本并压缩
$ make-big-lm -read ../counts/*.ngrams.gz -lm ../split.lm -order 3
#训练语言模型
```

## SRILM on CentOS
首先下载[srilm-1.7.1.tar.gz](http://www.speech.sri.com/projects/srilm/download.html)，然后顺序执行：
```
$ yum -y install gcc make git automake libtool autoconf gawk gzip bzip2 p7zip xz
$ tar -zxf srilm-1.7.1.tar.gz
$ pwd
/root/srilm
$ sbin/machine-type #检测机器类型
i686-m64
$ vim Makefile
# SRILM = /home/speech/stolcke/project/srilm/devel
    添加行：
SRILM = /root/srilm

MACHINE_TYPE := $(shell $(SRILM)/sbin/machine-type)
    替换为：
MACHINE_TYPE := i686-m64
$ vim common/Makefile.machine.机器类型 #我这里是“i686-m64”
NO_TCL = 1
    修改为：
NO_TCL = X

GAWK = /usr/bin/awk
    修改为：
GAWK = /usr/bin/gawk
$ make World #编译SRILM
$ vim /etc/profile
export SRILM_PATH=/root/srilm
export PATH=$PATH:$SRILM_PATH/bin:$SRILM_PATH/bin/i686-m64
$ source /etc/profile
$ make test #测试
```

## 参考资料：
- [Ngram统计语言模型](http://www.cnblogs.com/mengfanrong/p/4910400.html)
- [Ngram语言模型简单介绍](http://www.cnblogs.com/wzm-xu/p/4229819.html)
- [Ngram模型中文语料实验-分词与统计](http://www.xuebuyuan.com/91381.html)
- [我们是这样理解语言的/统计语言模型](http://www.flickering.cn/nlp/2015/02/%E6%88%91%E4%BB%AC%E6%98%AF%E8%BF%99%E6%A0%B7%E7%90%86%E8%A7%A3%E8%AF%AD%E8%A8%80%E7%9A%84-2%E7%BB%9F%E8%AE%A1%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B)