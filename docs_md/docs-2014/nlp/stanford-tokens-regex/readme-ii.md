title: Stanford TokensRegex(2)
date: 2016-11-07
tags: [NLP,CoreNLP,TokensRegex]
---
CoreNLP中用于在文本上定义模式并将其映射为语义对象的通用框架。它强调将文本描述为特征（词、标点等）序列，在这些特征上定义模式。非常像标准的正则表达式，不过是在特征级别工作，而不是字符级别。

<!--more-->
## TokensRegex实战：准备
首先得做些准备工作，比如特征序列：
```scala
val props = new java.util.Properties()
props.load(new java.io.FileInputStream("corenlp.properties"))
val pipeline = new StanfordCoreNLP(props)
val annotation = new Annotation("习近平和江泽民都当过中国国家主席。")
pipeline.annotate(annotation)
val tokens: java.util.List[CoreLabel] = annotation.get(classOf[CoreAnnotations.TokensAnnotation])
val sentents: java.util.List[CoreMap] = annotation.get(classOf[CoreAnnotations.SentencesAnnotation])
tokens.asScala.foreach(i => print(i.index(), i.word(), i.tag(), i.ner()))
```

运行程序输出如下：
```
(1,习近平,NR,PERSON)(2,和,CC,O)(3,江泽民,NR,PERSON)(4,都,AD,O)(5,当过,VV,O)(6,中国,NR,GPE)(7,国家,NN,O)(8,主席,NN,O)(9,。,PU,O)
```

`corenlp.properties`文件内容：
```
# Pipeline options
annotators = segment, ssplit, pos, ner
# segment
customAnnotatorClass.segment = edu.stanford.nlp.pipeline.ChineseSegmenterAnnotator
segment.model = C:/Users/jian/Desktop/znlp-data/stanford/edu/stanford/nlp/models/segmenter/chinese/ctb.gz
segment.sighanCorporaDict = C:/Users/jian/Desktop/znlp-data/stanford/edu/stanford/nlp/models/segmenter/chinese
segment.serDictionary = C:/Users/jian/Desktop/znlp-data/stanford/edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz
segment.sighanPostProcessing = true
# sentence split
ssplit.boundaryTokenRegex = [.]|[!?]+|[。]|[！？]+
# pos
pos.model = C:/Users/jian/Desktop/znlp-data/stanford/edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger
# ner
ner.model = C:/Users/jian/Desktop/znlp-data/stanford/edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz
ner.applyNumericClassifiers = false
ner.useSUTime = false
```

## TokensRegex实战：单个模式
```scala
val pattern: TokenSequencePattern = TokenSequencePattern.compile("([{ner:PERSON}] => \"one\") []{0,3} /当过/ ([{ner:GPE}] (?m){1,2} /国家主席/ => \"two\") => \"all\"")
val matcher: TokenSequenceMatcher = pattern.getMatcher(tokens)
while (matcher.find()) {
  print(s"\n\npattern: ${matcher.pattern().pattern()}")
  (0 to matcher.groupCount()).foreach(i => {
    println(s"\nmatched string: ${matcher.group(i)}, value: ${matcher.groupValue(i)}")
    matcher.groupNodes(i).asScala.foreach(j => print(j.get(classOf[CoreAnnotations.IndexAnnotation]), j.get(classOf[CoreAnnotations.TextAnnotation])))
  })
}
```

运行程序输出如下：
```
pattern: ([{ner:PERSON}] => "one") []{0,3} /当过/ ([{ner:GPE}] (?m){1,2} /国家主席/ => "two") => "all"
matched string: 习近平和江泽民都当过中国国家主席, value: STRING(all)
(1,习近平)(2,和)(3,江泽民)(4,都)(5,当过)(6,中国)(7,国家)(8,主席)
matched string: 习近平, value: STRING(one)
(1,习近平)
matched string: 中国国家主席, value: STRING(two)
(6,中国)(7,国家)(8,主席)(1,习近平,NR,PERSON,null)
```

## TokensRegex实战：多个模式
```scala
val pat1 = "(?$result [{ner:PERSON}|{ner:GPE}])"
val pat2 = "(?$result [{ner:GPE}] (?m){1,2} /国家主席/)"
val multiPattern: List[TokenSequencePattern] = TokenSequencePattern.compile(pat1) :: TokenSequencePattern.compile(pat2) :: Nil
val multiMatcher: MultiPatternMatcher[CoreMap] = TokenSequencePattern.getMultiPatternMatcher(multiPattern.asJava)
val multiResult: java.util.List[SequenceMatchResult[CoreMap]] = multiMatcher.findNonOverlapping(tokens)
multiResult.asScala.foreach(i => {
  print(s"\n\npattern: ${i.pattern().pattern()}")
  (0 to i.groupCount()).foreach(j => {
    println(s"\nmatched string: ${i.group(j)}")
    i.groupNodes(j).asScala.foreach(k => print(k.get(classOf[CoreAnnotations.IndexAnnotation]), k.get(classOf[CoreAnnotations.TextAnnotation])))
  })
})
```

运行程序输出如下：（`pattern.setPriority()`设置优先级）
```
pattern: (?$result [{ner:PERSON}|{ner:GPE}])
matched string: 习近平
(1,习近平)
matched string: 习近平
(1,习近平)

pattern: (?$result [{ner:PERSON}|{ner:GPE}])
matched string: 江泽民
(3,江泽民)
matched string: 江泽民
(3,江泽民)

pattern: (?$result [{ner:GPE}] (?m){1,2} /国家主席/)
matched string: 中国国家主席
(6,中国)(7,国家)(8,主席)
matched string: 中国国家主席
(6,中国)(7,国家)(8,主席)
```

`SequenceMatchResult`与`MatchedGroupInfo`方法的关联关系：
```
i.groupMatchResults() <=> i.groupInfo().matchResults
i.groupValue() <=> i.groupInfo().value // (?$name pattern => value) 捕获组的值
i.? <=> i.groupInfo().varName // (?$name pattern => value) 捕获组的名
i.group() <=> i.groupInfo().text
i.groupNodes() <=> i.groupInfo().nodes
```

## TokensRegex实战：更复杂的场景
分配规则用于将值分配给变量，以供以后在提取规则中使用或用于模式中的扩展。提取规则用于提取匹配正则表达式的文本或特征。提取规则分为多个`stage`，每个`stage`包括以下内容：

1. 执行`text`和`tokens`规则，规则直接应用于`CoreMap`的`text`或`tokens`字段
2. 执行`composite`规则，匹配的表达式被合并，并重复应用直到没有新的匹配
3. 执行`filter`规则，在最后阶段，最终过滤阶段过滤掉无效表达式

```scala
val props = new java.util.Properties()
props.load(new java.io.FileInputStream("corenlp.properties"))
val pipeline = new StanfordCoreNLP(props)
val annotation = new Annotation("为您提供奔驰？所有车型的图片。")
pipeline.annotate(annotation)
val env = TokenSequencePattern.getNewEnv // Extractor
val extractor = CoreMapExpressionExtractor.createExtractorFromFiles(env, "token-regex-rules.txt")
val sentences: java.util.List[CoreMap] = annotation.get(classOf[CoreAnnotations.SentencesAnnotation]) // work
sentences.asScala.foreach(sentence => extractor.extractExpressions(sentence).asScala.foreach(i => {
  println("value: " + i.getValue.get() + ", text: " + i.getText + ", priority: " + i.getPriority + ", order: " + i.getOrder + ", CharOffsets" + i.getCharOffsets + ", TokenOffsets" + i.getTokenOffsets)
  i.getAnnotation.get(classOf[CoreAnnotations.TokensAnnotation]).asScala.foreach(j => j.set(classOf[CoreAnnotations.NormalizedNamedEntityTagAnnotation], j.get(classOf[CoreAnnotations.NormalizedNamedEntityTagAnnotation]) + "++"))
}))
annotation.get(classOf[CoreAnnotations.TokensAnnotation]).asScala.foreach(i => println(i.index(), i.word(), i.tag(), i.ner(), i.get(classOf[CoreAnnotations.NormalizedNamedEntityTagAnnotation])))
```

运行程序输出如下：
```
value: RES_2, text: 奔驰, priority: 9.0, order: 0, CharOffsets(4,6), TokenOffsets(3,4)
value: RES_1, text: 车型, priority: 1.0, order: 1, CharOffsets(9,11), TokenOffsets(6,7)
value: RES_1, text: 图片, priority: 1.0, order: 2, CharOffsets(12,14), TokenOffsets(8,9)
(1,为,P,O,null)
(2,您,PN,O,null)
(3,提供,VV,O,提供奔驰)
(4,奔驰,VV,NE_2,提供奔驰++)
(5,？,PU,O,null)
(6,所有,DT,O,null)
(7,车型,NN,NE_1,车型++)
(8,的,DEG,O,null)
(9,图片,NN,NE_1,图片++)
(10,。,PU,O,null)
```

`token-regex-rules.txt`文件内容：
```
tokens = {type: "CLASS", value: "edu.stanford.nlp.ling.CoreAnnotations$TokensAnnotation"}
ner = {type: "CLASS", value: "edu.stanford.nlp.ling.CoreAnnotations$NamedEntityTagAnnotation"}
normalized = {type: "CLASS", value: "edu.stanford.nlp.ling.CoreAnnotations$NormalizedNamedEntityTagAnnotation"}
# 1
ENV.defaults["stage"] = 0
ENV.defaults["ruleType"] = "tokens"
{
    ruleType: "tokens",
    pattern: ( (/奔驰|车型|图片/ => "NE_1") ),
    action: ( Annotate($1, ner, $$1.value), Annotate($0, normalized, $$0.text) ),
    matchedExpressionGroup: 1,
    result: "RES_1",
    priority: 1
}
{
    ruleType: "tokens",
    pattern: ( /提供/ (/奔驰/ => "NE_2") ),
    action: ( Annotate($1, ner, $$1.value), Annotate($0, normalized, $$0.text) ),
    matchedExpressionGroup: 1,
    result: "RES_2",
    priority: 9
}
```

## TokensRegexAnnotator实战：使用管道
如果你只想使用TokensRegex用正则表达式识别命名实体，那么你应该用TokensRegexNERAnnotator代替。例如，添加`ner:COLOR`标签，并把RGB代码规范化：
```scala
val props = new java.util.Properties()
props.load(new java.io.FileInputStream("corenlp.properties"))
props.put("annotators", "segment, ssplit, pos, ner, color")
props.put("customAnnotatorClass.color", "edu.stanford.nlp.pipeline.TokensRegexAnnotator")
props.put("color.rules", "color.rules.txt")
val pipeline = new StanfordCoreNLP(props)
val annotation = new Annotation("Both blue and light blue are nice colors.")
pipeline.annotate(annotation)
val tokens: java.util.List[CoreLabel] = annotation.get(classOf[CoreAnnotations.TokensAnnotation])
tokens.asScala.foreach(i => print(i.index(), i.word(), i.tag(), i.ner(), i.get(classOf[CoreAnnotations.NormalizedNamedEntityTagAnnotation])))
```

运行程序输出如下：
```
(1,Both,NR,O,null)(2,blue,JJ,COLOR,#0000FF)(3,and,CC,O,null)(4,light,JJ,COLOR,#ADD8E6)(5,blue,JJ,COLOR,#ADD8E6)(6,are,VC,O,null)(7,nice,VA,O,null)(8,colors,VA,O,null)(9,.,PU,O,null)
```

`color.rules.txt`文件内容：
```
# Case insensitive pattern matching (see java.util.regex.Pattern flags)
ENV.defaultStringPatternFlags = 2
# Map variable names to annotation keys
tokens = { type: "CLASS", value: "edu.stanford.nlp.ling.CoreAnnotations$TokensAnnotation" }
ner = { type: "CLASS", value: "edu.stanford.nlp.ling.CoreAnnotations$NamedEntityTagAnnotation" }
normalized = { type: "CLASS", value: "edu.stanford.nlp.ling.CoreAnnotations$NormalizedNamedEntityTagAnnotation" }
# Create variable
$Colors = (
  /red/     => "#FF0000" |
  /green/   => "#00FF00" |
  /blue/    => "#0000FF" |
  (/pale|light/) /blue/   => "#ADD8E6"
)
# Define ruleType to be over tokens
ENV.defaults["ruleType"] = "tokens"
# Define rule
# annotate matched tokens ($0) with ner="COLOR" and normalized=matched value ($$0.value)
{
    pattern: ( $Colors ),
    action: ( Annotate($0, ner, "COLOR"), Annotate($0, normalized, $$0.value ) )
}
```

## 参考资料：
- [Stanford TokensRegex](http://nlp.stanford.edu/software/tokensregex.shtml)
- [Annotator dependencies](http://stanfordnlp.github.io/CoreNLP/dependencies.html)
- [TokensRegexAnnotator](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/pipeline/TokensRegexAnnotator.html)
- [TokenSequencePattern](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/TokenSequencePattern.html)
- [SequenceMatchRules](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/SequenceMatchRules.html)
- [CoreMapExpressionExtractor](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/CoreMapExpressionExtractor.html)