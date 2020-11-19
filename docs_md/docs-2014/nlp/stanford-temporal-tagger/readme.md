title: Stanford时间标注器
date: 2016-10-11
tags: [NLP,CoreNLP,Tagger]
---
Stanford Temporal Tagger(SUTime)是一个识别和规范化时间表达式的开源库。如`next wednesday at 3pm`转化为`2016-02-17T15:00`，根据假定当前基准时间。它是基于规则的可扩展实现，使用[Stanford TokensRegex](http://nlp.stanford.edu/software/tokensregex.shtml)开发。

<!--more-->
## 准备
你需要下载[code jar](http://stanfordnlp.github.io/CoreNLP/)和[models jar](http://nlp.stanford.edu/software/stanford-english-corenlp-2016-01-10-models.jar)。本文为了方便探索，使用以下相对繁琐的方式。直接拷贝源码[stanfordnlp/CoreNLP](https://github.com/stanfordnlp/CoreNLP)到新的Java/Scala工程中。手动配置core jar的依赖项：[lib](https://github.com/stanfordnlp/CoreNLP/tree/master/lib)
```
libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.21"
libraryDependencies += "org.slf4j" % "slf4j-simple" % "1.7.21"
libraryDependencies += "joda-time" % "joda-time" % "2.9"
libraryDependencies += "de.jollyday" % "jollyday" % "0.4.7"
libraryDependencies += "com.google.protobuf" % "protobuf-java" % "2.6.1"
libraryDependencies += "com.io7m.xom" % "xom" % "1.2.10"
libraryDependencies += "com.googlecode.efficient-java-matrix-library" % "ejml" % "0.23"
libraryDependencies += "javax.json" % "javax.json-api" % "1.0"
```

## 使用
SUTime依赖于分词、分句和词性标注。下面是一个完整例子：
```java
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.time.*;
import edu.stanford.nlp.util.CoreMap;
import java.util.List;
import java.util.Properties;
public class TestSUTime {
    public static void main(String[] args) {
        String basedir = "C:/Users/jian/Downloads/stanford-corenlp-3.6.0-models/edu/stanford/nlp/models/";
        Properties props = new Properties();
        props.setProperty("pos.model", basedir + "pos-tagger/english-left3words/english-left3words-distsim.tagger");
        props.setProperty("sutime.rules", basedir + "sutime/defs.sutime.txt," +
                basedir + "sutime/english.sutime.txt," +
                basedir + "sutime/english.holidays.sutime.txt");
        props.setProperty("sutime.binder.1.xml", basedir + "sutime/jollyday/Holidays_sutime.xml");
        props.setProperty("sutime.binder.1.pathtype", "file");

        AnnotationPipeline pipeline = new AnnotationPipeline();
        pipeline.addAnnotator(new TokenizerAnnotator(false, TokenizerAnnotator.TokenizerType.English));
        pipeline.addAnnotator(new WordsToSentencesAnnotator(false));
        pipeline.addAnnotator(new POSTaggerAnnotator(props.getProperty("pos.model"), false));
        pipeline.addAnnotator(new TimeAnnotator("sutime", props));

        String text = "Three interesting dates are 18 Feb 1997, the 20th of july and 4 days from today.";
        Annotation annotation = new Annotation(text);
        annotation.set(CoreAnnotations.DocDateAnnotation.class, "2013-07-14");//文档日期
        pipeline.annotate(annotation);
        System.out.println(annotation.get(CoreAnnotations.TextAnnotation.class));
        List<CoreMap> timexAnnsAll = annotation.get(TimeAnnotations.TimexAnnotations.class);
        for (CoreMap cm : timexAnnsAll) {
            List<CoreLabel> tokens = cm.get(CoreAnnotations.TokensAnnotation.class);
            System.out.println(cm + " [from char offset " +
                    tokens.get(0).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class) +
                    " to " + tokens.get(tokens.size() - 1).get(CoreAnnotations.CharacterOffsetEndAnnotation.class) + ']' +
                    " --> " + cm.get(TimeExpression.Annotation.class).getTemporal());
        }
        System.out.println("--");
    }
}
```

`basedir`指向的是`stanford-english-corenlp-2016-01-10-models.jar`解压后的目录。程序输出如下：
```
Three interesting dates are 18 Feb 1997, the 20th of july and 4 days from today.
18 Feb 1997 [from char offset 28 to 39] --> 1997-02-18
the 20th of july [from char offset 41 to 57] --> 2013-07-20
4 days from today [from char offset 62 to 79] --> 2013-07-18
--
```

遗憾的是SUTime只支持英文，对于中文没有现成可用的规则集。

## 参考资料：
- [Stanford Temporal Tagger](http://nlp.stanford.edu/software/sutime.shtml)
- [SUTime powerpoint slides](http://nlp.stanford.edu/software/SUTime.pptx)
- [stanfordnlp/CoreNLP](https://github.com/stanfordnlp/CoreNLP)