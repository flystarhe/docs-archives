title: Stanford TokensRegex(1)
date: 2016-11-02
tags: [NLP,CoreNLP,TokensRegex]
---
CoreNLP中用于在文本上定义模式并将其映射为语义对象的通用框架。它强调将文本描述为特征（词、标点等）序列，在这些特征上定义模式。非常像标准的正则表达式，不过是在特征级别工作，而不是字符级别。

<!--more-->
例如，你可能匹配作为画家的人的名字，使用TokensRagex模式如下：

    ([ner: PERSON]+) /was|is/ /an?/ []{0,3} /painter|artist/

**TokensRegex提供以下内容：**

1. 映射文本表达式为Java对象
2. 基于特征序列的模式匹配
3. 与CoreNLP管道集成

TokensRegex包的概述文件[doc1](http://nlp.stanford.edu/software/tokensregex/TokensRegexOverview.pdf)和[doc2](http://nlp.stanford.edu/pubs/tokensregex-tr-2014.pdf)，以及几个关键的类的详细文档：[TokenSequencePattern](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/TokenSequencePattern.html)和[Expressions](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/types/Expressions.html)。还有额外的幻灯片[下载](http://nlp.stanford.edu/software/tokensregex/TokensRegex.pdf)。

## TokensRegex Usage
CoreNLP管道提供了两个使用TokensRegex的annotators，它们可以被配置并添加到管道：

1. **TokensRegexNERAnnotator**：目标是提供一个简单的框架，其中包含未在传统NL语料库注解，但很容易被基于规则的技术认可的标签。类似RegexNERAnnotator，但支持TokensRegex表达式。[doc](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/pipeline/TokensRegexNERAnnotator.html)
2. **TokensRegexAnnotator**：使用TokensRegex规则定义模式匹配更通用的标注，比TokensRegexNERAnnotator更灵活，但用起来也更复杂。[doc](http://nlp.stanford.edu/software/tokensregex/tokensregexAnnotator.shtml)

TokensRegex遵循java正则表达式库类似的模式。规则使用字符串表示，然后编译成`TokenSequencePattern`，对于给定的特征序列创建`TokenSequenceMatcher`。这使得`TokenSequencePattern`将只编译一次。用法示例：
```java
List<CoreLabel> tokens = ...;
TokenSequencePattern pattern = TokenSequencePattern.compile(...);
TokenSequenceMatcher matcher = pattern.getMatcher(tokens);
while (matcher.find()) {
    String matchedString = matcher.group();
    List<CoreMap> matchedTokens = matcher.groupNodes();
    ...
}
```

`TokenSequenceMatcher`提供了类似`Matcher`获得结果的方法。此外，它还能访问特征`groupNodes`。有关完整列表，请参阅[SequenceMatchResult](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/SequenceMatchResult.html)。

通常情况下，你可能有许多正则表达式想匹配，`MultiPatternMatcher`是比依次匹配更高效的实现。用法示例：
```java
List<CoreLabel> tokens = ...;
List<TokenSequencePattern> tokenSequencePatterns = ...;
MultiPatternMatcher multiMatcher = TokenSequencePattern.getMultiPatternMatcher(tokenSequencePatterns);
// 使用指定的模式列表查找所有非重叠序列
// 交叠时，基于优先级、长度等选择匹配项
List<SequenceMatchResult<CoreMap>> multiMatcher.findNonOverlapping(tokens);
```

对于复杂的用例，TokensRegex提供了多正则表达式分阶段匹配的管道。它也提供了用于定义规则和表达式应如何匹配的语言。`CoreMapExpressionExtractor`从文件读取TokensRegex规则，并使用TokensRegex管道提取匹配：
```java
List<CoreMap> sentences = ...;
CoreMapExpressionExtractor extractor = CoreMapExpressionExtractor.createExtractorFromFiles(TokenSequencePattern.getNewEnv(), file1, file2,...);
for (CoreMap sentence:sentences) {
    List<MatchedExpression> matched = extractor.extractExpressions(sentence);
    ...
}
```

`TokensRegex`管道从文件中读取规则，并分`stage`应用这些规则。有四种类型的规则：`text, tokens, composite, filter`，每个`stage`提取规则如下：

1. 在每个阶段的开始，应用`text`和`tokens`规则抽取
2. 接着，反复应用`composite`规则直到没有检测到新的匹配
3. 在每个阶段的结束，`filter`规则应用于丢弃不应匹配的表达式

>每个子阶段内，交叠匹配是基于优先级、匹配长度和规则被指定的顺序来解决。

## TokensRegex Rules
有两种类型的规则：`assignment`规则定义供日后使用的变量，`extraction`规则用于匹配表达式。使用类JSON语言指定`extraction`规则。例如：
```
{
  // ruleType is "text", "tokens", "composite", or "filter"
  ruleType: "tokens",
  // pattern to be matched  
  pattern: ( ([{ ner:PERSON }]) /was/ /born/ /on/ ([{ ner:DATE }]) ),
  // value associated with the expression for which the pattern was matched
  // matched expressions are returned with "DATE_OF_BIRTH" as the value
  // (as part of the MatchedExpression class)
  result: "DATE_OF_BIRTH"
}
```

大多数字段是可选的，规则字段说明如下：[规则格式描述](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/SequenceMatchRules.html)
1. **ruleType**：规则类型，可选值`"tokens" | "text" | "composite" | "filter"`，默认`"tokens"`，必需字段
2. **pattern**：匹配模式，推荐阅读[TokenSequencePattern](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/TokenSequencePattern.html)和[Pattern](https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html?is-external=true)，必需字段
3. **action**：模式触发时要应用的操作列表，每个操作都是[TokensRegex Expression](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/types/Expressions.html)
4. **result**：结果值，表达式，推荐阅读[Expressions](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/types/Expressions.html)
5. **name**：标识规则名称，字符串
6. **stage**：规则将按照`stage`分组，从低到高顺序应用，整数
7. **active**：标识规则是否启用，逻辑值，默认`true`
8. **priority**：优先级，相同阶段优选高优先级规则的匹配，实数
9. **weight**：规则权重，实数（未使用）
10. **matchFindType**：可选值`FIND_NONOVERLAPPING | FIND_ALL`，默认`FIND_NONOVERLAPPING`
11. **matchWithResults**：是否返回匹配的结果，逻辑值，默认`false`
12. **matchedExpressionGroup**：标识将哪个组视为匹配的表达式组，整数，默认0

只有`pattern`与`result`字段的短规则可以书写为：

    { (([{ner:PERSON}]) /was/ /born/ /on/ ([{ner:DATE}])) => "DATE_OF_BIRTH" }

`extraction`规则有四种类型：

1. **text**：应用于原始文本，`pattern`字段形式为`/abc/`，`abc`为java正则表达式
2. **tokens**：应用于特征序列，`pattern`字段形式为`( abc )`，`abc`为TokensRegex表达式，默认类型
3. **composite**：应用于先前匹配的表达式`text, tokens or previous composite`，并重复应用直到没有新的匹配
4. **filter**：应用于先前匹配的表达式，它们在每个阶段结束时用于滤除表达式

`assignment`规则被用来定义供日后使用的变量。`assignment`规则在文件中：
```
ENV.defaults["stage"] = 0
ENV.defaults["ruleType"] = "tokens"
tokens = { type: "CLASS", value: "edu.stanford.nlp.ling.CoreAnnotations$TokensAnnotation" }
$DAYOFWEEK = "/monday|tuesday|wednesday|thursday|friday|saturday|sunday/"
$TIMEOFDAY = "/morning|afternoon|evening|night|noon|midnight/"
// Match expressions like "monday afternoon"
{
  ruleType: "tokens",
  pattern: ( $DAYOFWEEK $TIMEOFDAY ), 
  result: "TIME"
}
```

`assignment`规则在代码中：
```scala
val env = TokenSequencePattern.getNewEnv
env.bind("numtype", classOf[CoreAnnotations.NumericTypeAnnotation]) // using: Annotate($0, numtype, "NE_NUM1")
env.bind("$RELDAY1", "/today|yesterday|tomorrow|tonight|tonite/") // using: /it/ /was/ $RELDAY1
env.bind("$RELDAY2", TokenSequencePattern.compile(env, "/today|yesterday|tomorrow|tonight|tonite/")) // using: /it/ /was/ $RELDAY2
env.bind("::IS_OK", new TempNodePattern[CoreLabel]) // like: {word::IS_OK}, TempNodePattern extends NodePattern
```

更多用途详见[SequenceMatchRules](http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/ling/tokensregex/SequenceMatchRules.html)。

## Pattern Language
TokensRegex模式语言设计为类似标准java正则表达式，主要区别在匹配单个特征。CoreNLP中特征被表示为`CoreMap: Map(Class -> Object)`，TokensRegex支持通过指定`key`和要匹配的值匹配属性。每个特征由`[ <expression> ]`表示，`<expression>`指定属性如何匹配。

**基本表达式**：写在`{}`中，形式为`{ <attr1>; <attr2>.. }`。每个`<attr>`指定一个`<name> <matchfunc> <value>`。
1. 确切的字符串匹配`{ word:".." }`：如`[{ word:"cat" }]`匹配文本`cat`
2. 字符串正则表达式匹配`{ word:/../ }`：如`[{ word:/cat|dog/ }]`匹配文本`cat`或`dog`
3. 多个属性匹配`{ word:..; tag:.. }`：如`[{ word:/cat|dog/; tag:"NN" }]`匹配文本`cat`或`dog`，并且词性是`NN`
4. 数值表达式匹配`==, !=, >=, <=, >, <`：如`[{ word>=4 }]`匹配与具有数字值大于或等于`4`的文本

>作为简写，你可以用`"cat"`代替`[{ word:"cat" }]`，用`/cat|dog/`代替`[{ word:/cat|dog/ }]`

**复合表达式：**`!, &, |`。

1. `()`：分组表达式
2. `!{x}`：`[!{ tag:/VB.*/ }]`任何非动词特征
3. `{x} & {y}`：`[{word>=1000} & {word<=2000}]`特征是一个1000到2000之间的数字
4. `{x} | {y}`：`[{word::IS_NUM} | {tag:CD}]`特征是一个数字或词性为`CD`

**常用的`annotation`名称映射：**

1. word -> TextAnnotation
2. tag -> PartOfSpeechAnnotation
3. lemma -> LemmaAnnotation
4. ner -> NamedEntityTagAnnotation
5. normalized -> NormalizedNamedEntityTagAnnotation

**属性匹配表达式：**

1. `[]`：任何特征
2. `"abc"`：特征的文本精确匹配字符串`abc`
3. `/abc/`：特征的文本精确匹配正则表达式`abc`
4. `{key:"abc"}`：特征对应`annotation`精确匹配字符串`abc`
5. `{key:/abc/}`：特征对应`annotation`精确匹配正则表达式`abc`
6. `{key::IS_NUM}`：特征对应`annotation`是一个数字
7. `{key::IS_NIL} or {key::NOT_EXISTS}`：特征对应`annotation`不存在
8. `{key::NOT_NIL} or {key::EXISTS}`：特征对应`annotation`存在

**定义正则表达式匹配多个特征**：

    (?m){min,max} /pattern/

>特征序列的文本应与正则表达`pattern`匹配，特征数量介于min与max之间。例如`(?m){1,2} /国家主席/`。

**TokensRegex特征级的正则表达：**

1. `X Y`：`X`后续是`Y`
2. `X | Y`：`X`或者`Y`
3. `X & Y`：`X`并且`Y`
4. `(X)`：`X`作为捕获组
5. `(?$name X)`：`X`作为命名捕获组
6. `(?: X)`：`X`作为非捕获组

**TokensRegex包围字符总结：**

1. `[..]`：用来表示一个特征
2. `(..)`：用来表示分组（捕获组）
3. `{..}`：用来表示匹配的表达式（或次数）
4. `/../`：用来表示一个正则表达式
5. `".."`：用来表示一个字符串

## Test
这里假设你已有与[Stanford CoreNLP初探](#)相当的基础。这次考虑在Scala交互模式下进行探索，便于研究数据结构。先做些准备工作：
```
[root@c01 coreNLP]# pwd
/lab/coreNLP
[root@c01 coreNLP]# unzip stanford-chinese-corenlp-2016-01-19-models.jar -d data
[root@c01 coreNLP]# scala -cp stanford-corenlp-3.6.0.jar:*.jar -J-Xmx2g
scala> import scala.collection.JavaConverters._
scala> import edu.stanford.nlp.ling.tokensregex.{TokenSequenceMatcher, TokenSequencePattern}
scala> import edu.stanford.nlp.ling.{CoreLabel, CoreAnnotations}
scala> import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
scala> val basedir = "/lab/coreNLP/data/"
scala> val props = new java.util.Properties()
scala> props.setProperty("annotators", "segment, ssplit, pos, ner") // task
scala> props.setProperty("customAnnotatorClass.segment", "edu.stanford.nlp.pipeline.ChineseSegmenterAnnotator") // segment
scala> props.setProperty("segment.NormalizationTable", basedir + "edu/stanford/nlp/models/segmenter/chinese/norm.simp.utf8")
scala> props.setProperty("segment.normTableEncoding", "UTF-8")
scala> props.setProperty("segment.model", basedir + "edu/stanford/nlp/models/segmenter/chinese/ctb.gz")
scala> props.setProperty("segment.sighanCorporaDict", basedir + "edu/stanford/nlp/models/segmenter/chinese")
scala> props.setProperty("segment.serDictionary", basedir + "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz")
scala> props.setProperty("segment.sighanPostProcessing", "true")
scala> props.setProperty("ssplit.boundaryTokenRegex", "[.]|[!?]+|[。]|[！？]+") // sentence split
scala> props.setProperty("pos.model", basedir + "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger") // pos
scala> props.setProperty("ner.model", basedir + "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz") // ner
scala> props.setProperty("ner.applyNumericClassifiers", "false")
scala> props.setProperty("ner.useSUTime", "false")
scala> val pipeline = new StanfordCoreNLP(props)
```

OK，先做个简单的实验：
```
scala> val annotation = new Annotation("习近平和江泽民都当过中国国家主席。")
scala> pipeline.annotate(annotation)
scala> val tokens: java.util.List[CoreLabel] = annotation.get(classOf[CoreAnnotations.TokensAnnotation])
scala> tokens.asScala.foreach(i => print(i.index(), i.word(), i.tag(), i.ner()))
(1,习近平,NR,PERSON)(2,和,CC,O)(3,江泽民,NR,PERSON)(4,都,AD,O)(5,当过,VV,O)(6,中国,NR,GPE)(7,国家,NN,O)(8,主席,NN,O)(9,。,PU,O)
scala> val pattern: TokenSequencePattern = TokenSequencePattern.compile("([{ner:PERSON}]) []{0,3} /当过/ ([{ner:GPE}] /国家/ /主席/)")
scala> val matcher: TokenSequenceMatcher = pattern.getMatcher(tokens)
scala> while (matcher.find()) {
  (0 to matcher.groupCount()).foreach(i => {
    println(s"\nmatched string(group ${i}): ${matcher.group(i)}")
    matcher.groupNodes(i).asScala.foreach(j => print(j.get(classOf[CoreAnnotations.IndexAnnotation]), j.get(classOf[CoreAnnotations.TextAnnotation])))
  })
}
matched string(group 0): 习近平和江泽民都当过中国国家主席
(1,习近平)(2,和)(3,江泽民)(4,都)(5,当过)(6,中国)(7,国家)(8,主席)
matched string(group 1): 习近平
(1,习近平)
matched string(group 2): 中国国家主席
(6,中国)(7,国家)(8,主席)
```

## 参考资料：
- [Stanford TokensRegex](http://nlp.stanford.edu/software/tokensregex.shtml)