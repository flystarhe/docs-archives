title: Stanford RegexNER
date: 2016-11-01
tags: [NLP,CoreNLP,RegexNER]
---
在特征序列上使用java正则表达式的简单NER。它的目标是提供合并传统语料库中未标识的实体标签的简单框架。例如，模型文件中关于民族、宗教和标题的默认正则表达式列表。如何使用[RegexNER](http://nlp.stanford.edu/software/regexner)的简单例子。对于更复杂的应用，可以考虑使用[TokensRegex](http://nlp.stanford.edu/software/tokensregex.shtml)。

<!--more-->
## 一个例子
RegexNER是基于模式的易于使用的NER接口。这里有一个你可以用RegexNER做什么的例子。让我们从一个[小文件](readme01.txt)开始，其中包含维基百科的[习近平](https://zh.wikipedia.org/wiki/习近平)的信息。如果通过CoreNLP运行在命令行：

    java -mx2g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -props corenlp.properties -annotators 'segment,ssplit,pos,ner' -file 2016-11-01-01.txt

其中`corenlp.properties`内容如下：
```
# Pipeline options
annotators = segment, ssplit, pos, ner
# segment
customAnnotatorClass.segment = edu.stanford.nlp.pipeline.ChineseSegmenterAnnotator
segment.model = edu/stanford/nlp/models/segmenter/chinese/ctb.gz
segment.sighanCorporaDict = edu/stanford/nlp/models/segmenter/chinese
segment.serDictionary = edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz
segment.sighanPostProcessing = true
# sentence split
ssplit.boundaryTokenRegex = [.]|[!?]+|[。]|[！？]+
# pos
pos.model = edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger
# ner
ner.model = edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz
ner.applyNumericClassifiers = false
ner.useSUTime = false
```

## 尝试改进
输出为[XML文件](readme02.txt)，虽然不坏，但我们还是想改进它。例如，我们可能想要将`法学博士`标识为实体。通过RegexNER很容易实现。简单规则文件每行两个字段，以`Tab`分隔，第一个字段指示要匹配的文本，第二个字段具体要分配的类别。我们可能希望用`DEGREE`标识`法学博士`，第一个RegexNER文件：
```
法学 博士    DEGREE
```

我们可以在命令行使用这个文件，添加RegexNER到annotators：

    java -mx2g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -props corenlp.properties -annotators 'segment,ssplit,pos,ner,regexner' -regexner.mapping regexner.txt -file 2016-11-01-01.txt

## 使用正则语法
现在玩复杂一点，第一个字段不只是匹配一个字符串，而是作为一个序列的模式。CoreNLP将文本划分为特征序列，第一字段中空白分隔的模式必须与文本中的连续的特征匹配。每个模式都是标准的java正则表达式。如果正则表达式匹配一个标记序列，那么标记将被重新标记为第二列中的类别。对于上面的例子，你可能还想匹配各种不同学位：
```
(理学|法学) (学士|硕士|博士)    DEGREE
```

## 深入映射文件
如果你看看原始输出，你会看到有几个错误。这时你需要一个概念：RegexNER不会覆盖现有的实体分配，除非您在第三个制表符分隔的列中赋予它权限，其中包含可以覆盖的实体类型的逗号分隔列表。只有非实体`O`标签总是可以被覆盖，但你可以指定额外的实体标签，它们总是可以被覆盖：
```
(理学|法学) (学士|硕士|博士)    DEGREE
1953年 6月 15日    DATE    MISC
习明泽    PERSON
```

当然，上述重新标记的规则有时非常有用，不过有时也会变得危险。如果你想在上下文中检查单词时有更多控制，甚至检查词性。那你需要详细查看TokensRegex，它是一个更强大的框架，也更复杂。RegexNER比较简单，但通常足够前端使用。

你也不必使用RegexNER作为NER后处理器。例如，我们可以运行上面的RegexNER文件，像这样：

    java -mx2g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLP -props corenlp.properties -annotators 'segment,ssplit,regexner' -regexner.mapping regexner.txt -file 2016-11-01-01.txt

## RegexNERAnnotator
掌握了在命令行使用。你也可以很容易的在代码中使用RegexNER：
```java
Properties props = new java.util.Properties();
props.load(new java.io.FileInputStream("corenlp.properties"));
props.put("annotators", "segment, ssplit, pos, ner, regexner");
props.put("regexner.mapping", "your_path/regexner.txt");
StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
```

RegexNERAnnotator选项列表：

1. **regexner.ignoreCase**：如果为true，则会忽略大小写。默认false
2. **regexner.mapping**：逗号分隔要使用的映射文件列表。默认`edu/stanford/nlp/models/regexner/type_map_clean`
3. **regexner.mapping.header**：逗号分隔的字段列表（如果在映射文件中指定，则为true）。默认`pattern,ner,overwrite,priority,group`
4. **regexner.mapping.field.fieldname**：除ner字段外的字段的类映射。
5. **regexner.commonWords**：逗号分隔的文件列表，用于不需要注释的常用字词
6. **regexner.backgroundSymbol**：以逗号分隔的NER标签列表，始终替换。默认`O,MISC`
7. **regexner.posmatchtype**：如何使用`validpospattern`来匹配特征的POS。可选`MATCH_ALL_TOKENS`、`MATCH_AT_LEAST_ONE_TOKEN`和`MATCH_ONE_TOKEN_PHRASE_ONLY`，默认`MATCH_AT_LEAST_ONE_TOKEN`
8. **regexner.validpospattern**：用于匹配POS标记的正则表达式模式。
9. **regexner.noDefaultOverwriteLabels**：逗号分隔的输出类型列表，对于这些类型，仅当匹配的表达式具有与正则表达式指定的overwrite匹配的NER类型，NER类型才会被覆盖。
10. **regexner.verbose**：如果为true，则打开额外的调试消息。默认false

映射文件是制表符分隔文件。格式和输出字段可以通过指定不同的`regexner.mapping.header`来更改。例如，如果您想将`Stanford University`标记为`SCHOOL`的NER标记并链接到`https://en.wikipedia.org/wiki/Stanford_University`，则可以通过将标准化添加到标题中来实现：
```
regexner.mapping.header=pattern,ner,normalized,overwrite,priority,group
# Not needed, but illustrate how to link a field to an annotation
regexner.mapping.field.normalized=edu.stanford.nlp.ling.CoreAnnotations$NormalizedNamedEntityTagAnnotation
```

使您的映射文件具有以下条目：

    Stanford University\tSCHOOL\thttps://en.wikipedia.org/wiki/Stanford_University

然而，在一般情况下，写覆盖所有情况的规则是非常困难的。这也是统计分类占主导地位一个原因，因为善于整合各种证据来源。所以，RegexNER仅仅作为校正或补充统计自然语言处理系统等输出的工具。TokensRegex更强大，也更复杂。TokensRegex允许你有一个完整的规则文件库，更是模式允许在其他注释字段（如POS或NER）上进行匹配。

## TokensRegexNERAnnotator
此注释器设计为用作完整NER系统的一部分，以标记不属于通常的NER类别的实体。映射文件中参数是制表符分隔的，最后两个参数是可选的。在多个正则表达式与短语匹配的情况下，使用优先级等级（较高优先级）来在可能的类型之间进行选择。当优先级相同时，则较长匹配被优选。用户提供格式如下的文件：
```
regex1    TYPE    overwritableType1,Type2...    priority
regex2    TYPE    overwritableType1,Type2...    priority
...
```

第一列正则表达式可以采用以下两种格式之一：

1. TokensRegex表达式：以“(”开头，以“)”结尾，[语法](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/TokenSequencePattern.html)。
2. 正则表达式序列：由空格分隔，遵循java正则表达式约定。

与RegexNERAnnotator类似，但是使用TokensRegex作为匹配正则表达式的基础库。主要区别：

1. 当NER注释可以被覆盖时：如果找到的表达式与先前的NER短语重叠，那么NER标签不被替换；如果找到的表达式其中存在特征的NER标签不一致，那么NER标签被替换
2. 支持NamedEntityTagAnnotation字段之外的字段的注释
3. 同时支持TokensRegex模式和文本模式正则表达式
4. 由PosMatchType指定如何使用validpospattern
5. 默认，没有validPosPattern
6. 默认，O和MISC总是替换

选项列表参考RegexNERAnnotator，仅修改`regexner`为`tokenregexner`。实例中的`NormalizedNamedEntityTagAnnotation`也可以是实现了`CoreAnnotation`接口的用户类，代码如下：
```scala
val props = new java.util.Properties()
props.load(new java.io.FileInputStream("corenlp.properties"))
props.put("annotators", "segment, ssplit, pos, ner, tokenregexner")
props.put("customAnnotatorClass.tokenregexner", "edu.stanford.nlp.pipeline.TokensRegexNERAnnotator")
props.put("tokenregexner.mapping", "regexner.txt")
props.put("tokenregexner.mapping.header", "pattern,ner,overwrite,priority,user_tail")
props.put("tokenregexner.mapping.field.user_tail", "edu.stanford.nlp.ling.CoreAnnotations$NormalizedNamedEntityTagAnnotation")
val pipeline = new StanfordCoreNLP(props)
val annotation = new Annotation("习近平和江泽民都当过中国国家主席。")
pipeline.annotate(annotation)
val tokens: java.util.List[CoreLabel] = annotation.get(classOf[CoreAnnotations.TokensAnnotation])
tokens.asScala.foreach(i => print(i.index(), i.word(), i.tag(), i.ner(), i.get(classOf[CoreAnnotations.NormalizedNamedEntityTagAnnotation])))
```

运行程序输出如下：
```
// 关闭tokenregexner
(1,习近平,NR,PERSON,null)(2,和,CC,O,null)(3,江泽民,NR,PERSON,null)(4,都,AD,O,null)(5,当过,VV,O,null)(6,中国,NR,GPE,null)(7,国家,NN,O,null)(8,主席,NN,O,null)(9,。,PU,O,null)
// 开启tokenregexner
(1,习近平,NR,NE_P,tail_99)(2,和,CC,O,null)(3,江泽民,NR,NE_I,tail_99)(4,都,AD,NE_I,tail_99)(5,当过,VV,O,null)(6,中国,NR,NE_N,tail_90)(7,国家,NN,NE_N,tail_90)(8,主席,NN,NE_N,tail_90)(9,。,PU,O,null)
```

其中`regexner.txt`内容如下：
```
习近平 和 江泽民    NE_J    O    90    tail_90
江泽民 都    NE_I    O    99    tail_99
( [{ner:PERSON}] )    NE_P    O,PERSON    99    tail_99
( [{ner:GPE}] (?m){1,2} /国家主席/ )    NE_N    O    90    tail_90
```

## 参考资料：
- [Stanford RegexNER](http://nlp.stanford.edu/software/regexner.shtml)
- [RegexNERAnnotator](http://stanfordnlp.github.io/CoreNLP/regexner.html)
- [TokensRegexNERAnnotator](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/pipeline/TokensRegexNERAnnotator.html)
- [Annotator dependencies](http://stanfordnlp.github.io/CoreNLP/dependencies.html)