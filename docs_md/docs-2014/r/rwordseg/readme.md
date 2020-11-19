title: R分词之Rwordseg
date: 2015-05-20
tags: [R,Rwordseg]
---
Rwordseg是一个R环境下的中文分词工具，使用rJava调用Java分词工具Ansj。Ansj也是一个开源的Java中文分词工具，基于中科院的ictclas中文分词算法，采用隐马尔科夫模型。并支持行业字典、用户自定义字典。

<!--more-->
## 安装Rwordseg包
Rwordseg包依赖于rJava包和Java环境，在安装之前需要确保JRE已经安装，并且正确的设置设置了环境变量。

    > install.packages("rJava")
    > install.packages("Rwordseg", repos="http://R-Forge.R-project.org")

假设JRE安装在`D:/jre/jre1.8.0_45`，R的安装目录为`D:/R/R-3.0.3`，需要将以下路径添加到PATH环境变量中：

- D:/jre/jre1.8.0_45/bin
- D:/jre/jre1.8.0_45/bin/server
- D:/R/R-3.0.3/library/rJava/jri

>系统环境：windows 7 ultimate 64bit + R version 3.0.3 64bit + JRE 1.8.0 64bit。

## Rwordseg分词操作
**2.1 默认分词**
输入需要分词的句子(GBK或UTF-8编码都可以)，需要在R控制台中显示正常。

    > require(Rwordseg)
    载入需要的程辑包：Rwordseg
    载入需要的程辑包：rJava
    > segmentCN("结合成分子时")
    [1] "结合" "成" "分子" "时"

如果输入参数为字符向量，则返回列表。

    > segmentCN(c("说的确实在理", "一次性交多少钱"))
    [[1]]
    [1] "说" "的" "确实" "在" "理"
    [[2]]
    [1] "一次性" "交" "多少" "钱"

参数nosymbol表示是否只输出汉字、英文和数字，默认为TRUE，否则会输出标点符号。

    > segmentCN("想渊明、《停云》诗就，就此时风味。")
    [1] "想" "渊" "明" "停" "云" "诗" "就" "就" "此时" "风味"
    > segmentCN("想渊明、《停云》诗就，就此时风味。", nosymbol=FALSE)
    [1] "想" "渊" "明" "、" "《" "停" "云" "》" "诗" "就"
    [11] "，" "就" "此时" "风味" "。"

**2.2 词性识别**
参数nature可以设置是否输出词性，默认不输出，如果选择输出，那么返回的向量名为词性的标识。

    > segmentCN("花一元钱买了一朵美丽的花", nature=TRUE)
    v      m      n      v      ul      m      a      uj      v
    "花" "一元"   "钱"   "买"   "了" "一朵" "美丽"   "的"   "花"

不过目前的词性识别并不是真正意义上的智能词性识别，同一个词在语义上的词性还没有办法办法识别出来，结果仅作为参考。

**2.3 人名识别**
isNameRecognition选项可以设置是否进行智能的人名识别。智能识别有时候会和自定义的词库冲突，因此默认的选型是不进行人名识别。

    > getOption("isNameRecognition") # 查看默认设置
    [1] FALSE
    > segmentCN("梅野石是东昆仑盟主")
    [1] "梅"   "野"   "石"   "是"   "东"   "昆仑" "盟主"
    > segment.options(isNameRecognition=TRUE) # 启用人名识别
    > segmentCN("梅野石是东昆仑盟主")
    [1] "梅野石" "是" "东昆仑" "盟主"
    > getOption("isNameRecognition")
    [1] TRUE
    > segment.options(isNameRecognition=FALSE) # 还原默认设置

**2.4 tm格式支持**
segmentCN默认输出向量和列表，并使用向量的name属性来表示词性，这是R中最常用的数据结构。但是由于tm包已经成了R中事实的文本挖掘标准，因此常常会需要使用tm中使用空格分隔的单字符串格式。

returnType参数如果设置成“tm”，则表示输出tm格式的字符串，该方式无法输出词性。

isfast参数可以设定直接调用JAVA包进行最基础的分词，速度比较快，只能返回“tm”格式的文本，无法输出繁体字，也不能进行词性识别。

**2.5 对文件分词**
输入参数strwords除了可以是需要分词的字符向量，也可以是某个文本文件的路径。本包会自动判断是否为某个文件的路径，并自动识别字符编码。全部转换成UTF-8进行处理和输出。

如果输入文本路径，可以使用outfile参数指定输出文件的名称和路径，默认是与输入文件文件同路径并在原文件名基础上添加“segment”。blocklines表示每次读入的行数，默认是1000行，当输入文件比较大的时候可以根据电脑的性能来设置该参数。

    > segmentCN("说岳全传_GBK.txt")
    Output file: D:/说岳全传_GBK.segment.txt
    [1] TRUE

## Rwordseg词典管理
**3.1 安装和卸载**
该包支持安装新的词典，一次安装之后，每次重启R包时都会自动加载。目前支持普通格式的“text”文本词典和Sogou的“scel”格式细胞词典。

函数installDict用来安装新词典，参数dictpath表示词典的路径，参数dictname表示自定义的词典名称(建议用英文)，参数dicttype表示词典的类型(支持“text”和“scel”，默认为“text”)，参数load表示安装后是否自动加载到内存(默认是TRUE)。

普通文本格式词典的后缀名没有限制，ANSI和UTF-8的文件都可以支持，需要确保文件用编辑器打开时能正常显示。我们可以安装ansj项目中的[自定义词库](https://github.com/NLPchina/ansj_seg/blob/master/library/default.dic)。

    > listDict() # 查看已安装词典
    [1] Name Type Des Path
    <0 行> (或0-长度的row.names)
    > segmentCN("湖北大鼓真是不错啊")
    [1] "湖北" "大鼓" "真" "是" "不错" "啊"
    > installDict("E:/default.dic", dictname="ansj") # 安装新的词典
    386211 words were loaded! ... New dictionary 'ansj' was installed!
    > listDict() # 查看已安装词典
      Name Type Des
    1 ansj Ansj default.dic
      Path
    1 D:/Program Files/R/R-3.0.3/library/Rwordseg/dict/ansj.dic
    > segmentCN("湖北大鼓真是不错啊")
    [1] "湖北大鼓" "真是" "不错" "啊"
    > uninstallDict() # 卸载某些词典，默认卸载所有
    386211 words were removed! ... The dictionary 'ansj' was uninstalled!

这个例子只是展示如何加载这种格式的词典。例子中是一个巨大的词库，平常的使用中并不建议一次添加一个巨大的词库，因为分词的模型本来就是可以对新词进行划分，词库太大的效果并不一定好，而且加载时比较浪费时间。

在这里强烈推荐Sogou的[细胞词库](http://pinyin.sogou.com/dict/)，包含数目非常巨大的分类词典，而且可以找到最新的流行词。建议在进行具体分析研究之前找到相关的专业词典进行安装，比如要研究金庸的小说，可以下载词典“金庸武功招式.scel”。仍然用installDict来安装。

    > segmentCN("真武七截阵和天罡北斗阵那个厉害") # 未安装词典
    [1] "真" "武" "七" "截" "阵" "和" "天罡" "北斗" "阵" "那个"
    [11] "厉害"
    > installDict("E:/金庸武功招式.scel", dictname="jinyong") # 安装新的词典
    932 words were loaded! ... New dictionary 'jinyong' was installed!
    > segmentCN("真武七截阵和天罡北斗阵那个厉害") # 安装新词典后
    [1] "真武七截阵" "和" "天罡北斗阵" "那个" "厉害"
    > listDict() # 查看已安装词典
      Name    Type Des
    1 jinyong 读书 金庸小说中的武功和招式名称
      Path
    1 D:/Program Files/R/R-3.0.3/library/Rwordseg/dict/jinyong.dic
    > uninstallDict() # 卸载某些词典，默认卸载所有
    932 words were removed! ... The dictionary 'jinyong' was uninstalled!
    > listDict() # 查看已安装词典
    [1] Name Type Des Path
    <0 行> (或0-长度的row.names)

**3.2 自定义文本词典**
如果仅仅只是用户自己添加词汇，没有必要做一个词典进行安装，可以使用自定义词典的功能。默认的词典目录为`%R_HOME%/library/Rwordseg/dict`。可以在其中添加任意后缀为`.dic`的文本，里面可以输入自定义的词，每一行一个词，回车换行。可以直接写在`example.dic`这个文件，或者参考该文件新建dic文件。修改后每次重启时都会导入用户自定义词典。如果想要立即生效，可以在修改后运行`loadDict()`。

**3.3 手动添加和删除词汇**
如果仅仅只是在内存中临时添加或者删除词汇，可以使用insertWords和deleteWords。

    > segmentCN("画角声断焦门")
    [1] "画" "角" "声" "断" "焦" "门"
    > insertWords(c("画角","焦门")) # 临时添加词汇
    > segmentCN("画角声断焦门")
    [1] "画角" "声" "断" "焦门"
    > deleteWords(c("焦门")) # 临时删除词汇
    > segmentCN("画角声断焦门")
    [1] "画角" "声" "断" "焦" "门"

如果需要将添加或删除的词汇记录下来，每次重启时都会自动添加或删除这些词，可以将save参数设置为TRUE，例如：

    > insertWords("焦门", save=TRUE)
    > deleteWords(c("画角","焦门"), save=TRUE)

## 参考资料：
- [Rwordseg](http://jliblog.com/app/rwordseg)
- [解惑rJava:R与Java的高速通道](http://blog.fens.me/r-rjava-java/)