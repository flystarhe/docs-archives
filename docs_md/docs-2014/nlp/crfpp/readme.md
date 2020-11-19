title: CRF++实现中文分词
date: 2016-07-13
tags: [NLP,CRF++,分词]
---
条件随机场被用于中文分词和词性标注等词法分析工作，一般序列分类模型常常采用隐马尔可夫模型(HMM)。但隐马尔可夫模型中存在两个假设：输出独立性假设和马尔可夫性假设。条件随机场则使用一种概率图模型，具有表达长距离依赖性和交叠性特征的能力，能够较好地解决标注(分类)偏置等问题的优点，而且所有特征可以进行全局归一化，能够求得全局的最优解。

<!--more-->
## 安装CRF++
这里只介绍Linux下CRF++的安装和使用，并假设gcc(>3.0)编译器正确安装。[crf++ source](http://code.google.com/p/crfpp/downloads/list)

    $ tar -zxf CRF++-0.58.tar.gz
    $ cd CRF++-0.58
    $ ./configure
    $ make
    $ make install

考虑使用python工具包进行训练和测试，需要安装python工具包。进入python文件夹，运行以下命令安装：

    $ cd python
    $ python setup.py build
    $ python setup.py install

可以在python解释器下测试，是否能成功`import CRFPP`。若遇`ImportError: libcrfpp.so.0: cannot open shared object file: No such file or directory`则执行命令：

    32位系统：ln -s /usr/local/lib/libcrfpp.so.* /usr/lib/
    64位系统：ln -s /usr/local/lib/libcrfpp.so.* /usr/lib64/

## 训练文件和测试文件格式
训练文件和测试文件都要符合特定的格式CRF++才能正常工作。训练文件和测试文件都包含很多tokens(记录)行，一个记录由固定数目的列组成，列之间用空格或Tab间隔。[多个连续的token组成一个句子，为了识别句子的边界，会加入一个空行。]()

你可以在一个记录中设置很多列，但是所有记录的列数必须相同。在列中有一些常用的语义，例如第一列是字，第二列是字的标签，第三列是子标签等等。在`example/seg`文件夹里面，可以看到四个文件：exec.sh（执行脚本）、template（特征模板）、test.data（测试集）、train.data（训练集)。

    $ head train.data
    毎   k   B
    日   k   I
    新   k   I
    聞   k   I
    社   k   I
    特   k   B
    別   k   I
    顧   k   B
    問   k   I
    ４   n   B

这里第一列是待分词的日文字，第二列暂且认为是词性标记，第三列是字标注中的2-tag(B, I)标记，这个很重要，对于我们需要准备的训练集，主要是把这一列的标记做好，不过需要注意的是，其断句是靠空行来完成的。

    $ head test.data
    よ   h   I
    っ   h   I
    て   h   I
    私   k   B
    た   h   B
    ち   h   I
    の   h   B
    世   k   B
    代   k   I
    が   h   B

同样也有3列，第一列是日文字，第二列第三列与上面是相似的，不过在测试集里第三列主要是占位作用。事实上，CRF++对于训练集和测试集文件格式的要求是比较灵活的，首先需要多列，但不能不一致，既在一个文件里有的行是两列，有的行是三列；其次第一列代表的是需要标注的“字或词”，最后一列是输出位“标记tag”，如果有额外的特征，例如词性什么的，可以加到中间列里，所以训练集或者测试集的文件最少要有两列。
```
$ cat example/seg/template //example/basenp/template也很有参考价值
# Unigram
U00:%x[-2,0]
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U04:%x[2,0]
U05:%x[-2,0]/%x[-1,0]/%x[0,0]
U06:%x[-1,0]/%x[0,0]/%x[1,0]
U07:%x[0,0]/%x[1,0]/%x[2,0]
U08:%x[-1,0]/%x[0,0]
U09:%x[0,0]/%x[1,0]

# Bigram
B

```

关于CRF++中特征模板的说明和举例，请大家参考官方文档上的“Preparing feature templates”这一节，不想有意外请保持空行。而以下部分的说明拿上述日文分词数据举例。在特征模板文件中，每一行(如U00:%x[-2,0]）代表一个特征，而宏“%x[行位置,列位置]”则代表了相对于当前指向的token的行偏移和列的绝对位置，以上述训练集为例，如果当前扫描到“新 k I”这一行：

    毎   k   B
    日   k   I
    新   k   I <== 扫描到这一行，代表当前位置
    聞   k   I
    社   k   I
    特   k   B
    別   k   I
    顧   k   B
    問   k   I
    ４   n   B

那么依据特征模板文件抽取的特征如下：
```
# Unigram
U00:%x[-2,0]   ==> 毎
U01:%x[-1,0]   ==> 日
U02:%x[0,0]    ==> 新
U03:%x[1,0]    ==> 聞
U04:%x[2,0]    ==> 社
U05:%x[-2,0]/%x[-1,0]/%x[0,0]   ==> 毎/日/新
U06:%x[-1,0]/%x[0,0]/%x[1,0]    ==> 日/新/聞
U07:%x[0,0]/%x[1,0]/%x[2,0]     ==> 新/聞/社
U08:%x[-1,0]/%x[0,0]            ==> 日/新
U09:%x[0,0]/%x[1,0]             ==> 新/聞

# Bigram
B

```

CRF++里将特征分成两种类型，一种是Unigram的，“U”起头，另外一种是Bigram的，“B”起头。对于Unigram的特征，假如一个特征模板是`U01:%x[-1,0]`，CRF++会自动的生成一组特征函数`(func1 .. funcN)`集合。生成的特征函数的数目`= (L * N)`，其中L是输出的类型的个数，这里是B、I这两个tag，N是通过模板扩展出来的所有单个字符串(特征)的个数，这里指的是在扫描所有训练集的过程中找到的日文字。而Bigram特征主要是当前的token和前面一个位置token的自动组合生成的bigram特征集合。

    $ cat exec.sh
    #!/bin/sh
    ../../crf_learn -a CRF-L2 -c 4.0 -f 3 -p 4 template train.data model
    ../../crf_test -m model test.data
    ../../crf_learn -a MIRA -f 3 template train.data model
    ../../crf_test -m model test.data
    rm -f model

执行脚本告诉了我们如何训练一个CRF模型，以及如何利用这个模型来进行测试，执行这个脚本之后，对于输入的测试集，输出结果多了一列。而这一列才是模型预测的该字的标记tag，也正是我们所需要的结果。参数说明：

    crf_learn:
    -f, --freq=INT   使用属性的出现次数不少于INT(默认为1)
    -m, --maxiter=INT   设置INT为LBFGS的最大跌代次数(默认10k)
    -c, --cost=FLOAT   设置FLOAT为代价参数，过大会过度拟合(默认1.0)
    -e, --eta=FLOAT   设置终止标准FLOAT(默认0.0001)
    -C, --convert   将文本模式转为二进制模式
    -t, --textmodel   为调试建立文本模型文件
    -a, --algorithm=(CRF|MIRA)   选择训练算法，CRF-L2 or CRF-L1，默认为CRF-L2
    -p, --thread=INT   线程数(默认1)，利用多个CPU减少训练时间
    -H, --shrinking-size=INT   设置INT为最适宜的跌代变量次数(默认20)
    -v, --version   显示版本号并退出
    -h, --help   显示帮助并退出
    crf_test:
    $ crf_test -v1 -m model test.data| head
    # 0.478113
    Rockwell        NNP     B       B/0.992465
    International   NNP     I       I/0.979089
    Corp.   NNP     I       I/0.954883
    's      POS     B       B/0.986396
    Tulsa   NNP     I       I/0.991966

## 利用CRF++实现中文分词
首先将[backoff2005](http://www.sighan.org/bakeoff2005/)里的训练数据转化为CRF++所需的训练数据格式，以微软亚洲研究院提供的中文分词语料为例，依然采用4-tag(B(Begin，词首), E(End，词尾), M(Middle，词中), S(Single，单字词))标记集，只处理utf-8编码文本。原始训练集`./icwb2-data/training/msr_training.utf8`的形式是人工分好词的中文句子形式，如：

    心  静  渐  知  春  似  海  ，  花  深  每  觉  影  生  香  。
    吃  屎  的  东西  ，  连  一  捆  麦  也  铡  不  动  呀  ？

这里提供一个脚本[make_crf_train_data.py](https://github.com/ictlyh/CRFSegment/blob/master/make_crf_train_data.py)，将这个训练语料转换为CRF++训练用的语料格式(2列，4-tag)：

```
#!/usr/bin/python
# -*- coding: utf-8 -*-
#make_crf_train_data.py
#得到CRF++要求的格式的训练文件
#用法：命令行--python dataprocess.py input_file output_file

import sys
import codecs

#4 tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
def character_4tagging(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "\tS\n")
            else:
                output_data.write(word[0] + "\tB\n")
                for w in word[1:len(word)-1]:
                    output_data.write(w + "\tM\n")
                output_data.write(word[len(word)-1] + "\tE\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()

#6 tags for character tagging: B(Begin), E(End), M(Middle), S(Single), M1, M2
def character_6tagging(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "\tS\n")
            elif len(word) == 2:
                output_data.write(word[0] + "\tB\n")
                output_data.write(word[1] + "\tE\n")
            elif len(word) == 3:
                output_data.write(word[0] + "\tB\n")
                output_data.write(word[1] + "\tM\n")
                output_data.write(word[2] + "\tE\n")
            elif len(word) == 4:
                output_data.write(word[0] + "\tB\n")
                output_data.write(word[1] + "\tM1\n")
                output_data.write(word[2] + "\tM\n")
                output_data.write(word[3] + "\tE\n")
            elif len(word) == 5:
                output_data.write(word[0] + "\tB\n")
                output_data.write(word[1] + "\tM1\n")
                output_data.write(word[2] + "\tM2\n")
                output_data.write(word[3] + "\tM\n")
                output_data.write(word[4] + "\tE\n")
            elif len(word) > 5:
                output_data.write(word[0] + "\tB\n")
                output_data.write(word[1] + "\tM1\n")
                output_data.write(word[2] + "\tM2\n")
                for w in word[3:len(word)-1]:
                    output_data.write(w + "\tM\n")
                output_data.write(word[len(word)-1] + "\tE\n")
        output_data.write("\n")
    input_data.close()
    output_data.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: python dataprocess.py inputfile outputfile"
        sys.exit()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    character_4tagging(input_file, output_file)
```

只需要执行以下命令就可以得到CRF++要求的格式的训练文件`4tag_train_data.utf8`：

    python make_crf_train_data.py ./icwb2-data/training/msr_training.utf8 crf_4tag_train_data.utf8

有了这份训练语料，就可以利用crf的训练工具crf_learn来训练模型了，执行如下命令即可：

    ./CRF/crf_learn -f 3 -c 4.0 ./CRF/example/seg/template crf_4tag_train_data.utf8 crf_model

耗时稍微有些长，最终训练的model约51M。有了模型，现在我们需要做得还是准备一份CRF++用的测试语料，然后利用CRF++的测试工具crf_test进行字标注。原始的测试语料是`icwb2-data/testing/msr_test.utf8`，如：

    扬帆远东做与中国合作的先行
    希腊的经济结构较特殊。

下面我们用一个python脚本进行分词测试，测试脚本[crf_segment.py](https://github.com/ictlyh/CRFSegment/blob/master/crf_segmenter.py)如下：

```
#!/usr/bin/python
# -*- coding: utf-8 -*-
#crf_segmenter.py
#Usage:python crf_segmenter.py crf_model test_file result_file
# 利用CRF自带的python工具包，对输入文本进行分词

import codecs
import sys
import CRFPP

def crf_segmenter(input_file, output_file, tagger):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        tagger.clear()
        for word in line.strip():
            word = word.strip()
            if word:
                tagger.add((word + "\to\tB").encode('utf-8'))
        tagger.parse()
        size = tagger.size()
        xsize = tagger.xsize()
        for i in range(0, size):
            for j in range(0, xsize):
                char = tagger.x(i, j).decode('utf-8')
                tag = tagger.y2(i)
                if tag == 'B':
                    output_data.write(' ' + char)
                elif tag == 'M':
                    output_data.write(char)
                elif tag == 'E':
                    output_data.write(char + ' ')
                else: #tag == 'S'
                    output_data.write(' ' + char + ' ')
        output_data.write('\n')
    input_data.close()
    output_data.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: python crf_segmenter.py crf_model test_file result_file"
        sys.exit()
    crf_model = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    tagger = CRFPP.Tagger("-m " + crf_model)
    crf_segmenter(input_file, output_file, tagger)
```

运行以下命令进行分词：

    python crf_segmenter.py crf_model ./icwb2-data/testing/msr_test.utf8 crf_4tag_result.utf8

最终得到的分词结果事例如下：

    扬帆  远东  做  与  中国  合作  的  先行
    希腊  的  经济  结构  较  特殊  。

用backoff2005自带的测试脚本来检查分词效果：

    perl ./icwb2-data/scripts/score ./icwb2-data/gold/msr_training_words.utf8 ./icwb2-data/gold/msr_test_gold.utf8 crf_4tag_result.utf8 > crf_4tag_score.txt

评分结果如下：

    === SUMMARY:
    === TOTAL INSERTIONS:   1421
    === TOTAL DELETIONS:    1276
    === TOTAL SUBSTITUTIONS:    2412
    === TOTAL NCHANGE:  5109
    === TOTAL TRUE WORD COUNT:  106873
    === TOTAL TEST WORD COUNT:  107018
    === TOTAL TRUE WORDS RECALL:    0.965
    === TOTAL TEST WORDS PRECISION: 0.964
    === F MEASURE:  0.965
    === OOV Rate:   0.026
    === OOV Recall Rate:    0.647
    === IV Recall Rate: 0.974

## 参考资料：
- [CRF++实现中文分词](http://www.mutouxiaogui.cn/blog/?p=224)
- [定制你自己的CRF模型](https://github.com/NLPchina/ansj_seg/wiki/定制你自己的CRF模型)
- [CRF++: Yet Another CRF toolkit](https://taku910.github.io/crfpp/)