title: Stanford CoreNLP初探
date: 2016-09-27
tags: [NLP,CoreNLP,分词,词性标注,实体识别]
---
CoreNLP是Stanford提供的一套自然语言分析工具。对于用户提供的一段文字，无论是公司名、人名，还是时间日期、数量，它都能提供出每个词语的组成与语法，并且用短语、词汇间的依赖关系来标记出语句的组成结构，比如那些名词指代同一个事物，比如根据上下文的描述来分析观点与开放式的事物关系等。

<!--more-->
## 准备
你需要下载[code jar](http://stanfordnlp.github.io/CoreNLP/)和[models jar](http://stanfordnlp.github.io/CoreNLP/)，后者应该根据你使用的语言来选择。而code jar会依赖其它jar，怕麻烦的朋友可以使用maven或sbt：
```
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0"
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" classifier "models-chinese"
```

本文为了方便探索，使用以下相对繁琐的方式。直接拷贝源码[stanfordnlp/CoreNLP](https://github.com/stanfordnlp/CoreNLP)到新的Java/Scala工程中。另：下文中`basedir`指向的就是models jar解压后的目录。这种方式需要手动配置core jar的依赖项：[lib](https://github.com/stanfordnlp/CoreNLP/tree/master/lib)
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

## 分词
```java
public class TestSeg {
    public Annotation worker(Annotation annotation) throws Exception {
        String basedir = "C:/Users/jian/Downloads/stanford-chinese-corenlp-2015-12-08-models/";
        Properties props = new Properties();
        props.setProperty("NormalizationTable", basedir + "edu/stanford/nlp/models/segmenter/chinese/norm.simp.utf8");
        props.setProperty("normTableEncoding", "UTF-8");
        props.setProperty("sighanCorporaDict", basedir + "edu/stanford/nlp/models/segmenter/chinese");
        props.setProperty("serDictionary", basedir + "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz");
        props.setProperty("sighanPostProcessing", "true");
        CRFClassifier<CoreLabel> segmenter = new CRFClassifier<CoreLabel>(props);
        segmenter.loadClassifierNoExceptions(basedir + "edu/stanford/nlp/models/segmenter/chinese/pku.gz", props);

        String text = annotation.get(CoreAnnotations.TextAnnotation.class);
        List<CoreLabel> tokens = new ArrayList<>();
        int pos = 0;
        for (String w : segmenter.segmentString(text)) {
            if (w.isEmpty()) {
                continue;
            }
            CoreLabel token = new CoreLabel();
            token.setWord(w);
            token.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class, pos);
            token.set(CoreAnnotations.CharacterOffsetEndAnnotation.class, pos + w.length());
            tokens.add(token);
            pos += w.length();
        }
        annotation.set(CoreAnnotations.TokensAnnotation.class, tokens);
        return annotation;
    }

    public static void main(String[] args) throws Exception {
        String text = "何剑你好，听说你去中国人民大学参观了！四川省政府和腾讯成都分公司领导在青岛开会。";
        Annotation annotation = new Annotation(text);
        new TestSeg().worker(annotation);
        List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
        for (CoreLabel token : tokens) {
            System.out.println(token.beginPosition() + "_" + token.word());
        }
    }
}
```

## 分句
```java
public class TestSent {
    public Annotation worker(Annotation annotation) throws Exception {
        new TestSeg().worker(annotation);

        String splitRegex = "[.]|[!?]+|[。]|[！？]+";
        WordToSentenceProcessor<CoreLabel> wts = new WordToSentenceProcessor<CoreLabel>(splitRegex, null, null, WordToSentenceProcessor.NewlineIsSentenceBreak.NEVER, null, null);

        String text = annotation.get(CoreAnnotations.TextAnnotation.class);
        List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
        List<CoreMap> sentences = new ArrayList<>();
        int tokenOffset = 0;
        for (List<CoreLabel> sentenceTokens : wts.process(tokens)) {
            if (sentenceTokens.isEmpty()) {
                continue;
            }
            int begin = sentenceTokens.get(0).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
            int end = sentenceTokens.get(sentenceTokens.size() - 1).get(CoreAnnotations.CharacterOffsetEndAnnotation.class);

            Annotation sentence = new Annotation(text.substring(begin, end));
            sentence.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class, begin);
            sentence.set(CoreAnnotations.CharacterOffsetEndAnnotation.class, end);
            sentence.set(CoreAnnotations.TokensAnnotation.class, sentenceTokens);
            sentence.set(CoreAnnotations.TokenBeginAnnotation.class, tokenOffset);
            sentence.set(CoreAnnotations.TokenEndAnnotation.class, tokenOffset + sentenceTokens.size());
            sentence.set(CoreAnnotations.SentenceIndexAnnotation.class, sentences.size());
            tokenOffset += sentenceTokens.size();

            int index = 1;
            for (CoreLabel token : sentenceTokens) {
                token.setIndex(index++);
                token.setSentIndex(sentences.size());
            }
            sentences.add(sentence);
        }
        annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
        return annotation;
    }

    public static void main(String[] args) throws Exception {
        String text = "何剑你好，听说你去中国人民大学参观了！四川省政府和腾讯成都分公司领导在青岛开会。";
        Annotation annotation = new Annotation(text);
        new TestSent().worker(annotation);
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            Integer sentInd = sentence.get(CoreAnnotations.SentenceIndexAnnotation.class);
            for (CoreLabel token : tokens) {
                System.out.println(sentInd + "_" + token.beginPosition() + "_" + token.word());
            }
        }
    }
}
```

## POS标注
```java
public class TestPos {
    public Annotation worker(Annotation annotation) throws Exception {
        new TestSent().worker(annotation);

        String basedir = "C:/Users/jian/Downloads/stanford-chinese-corenlp-2015-12-08-models/";
        MaxentTagger tagger = new MaxentTagger(basedir + "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger");

        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            List<TaggedWord> tagged = tagger.tagSentence(tokens);
            if (tagged != null) {
                for (int i = 0, sz = tokens.size(); i < sz; i++) {
                    tokens.get(i).set(CoreAnnotations.PartOfSpeechAnnotation.class, tagged.get(i).tag());
                }
            } else {
                for (CoreLabel token : tokens) {
                    token.set(CoreAnnotations.PartOfSpeechAnnotation.class, "X");
                }
            }
        }
        return annotation;
    }

    public static void main(String[] args) throws Exception {
        String text = "何剑你好，听说你去中国人民大学参观了！四川省政府和腾讯成都分公司领导在青岛开会。";
        Annotation annotation = new Annotation(text);
        new TestPos().worker(annotation);
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            Integer sentInd = sentence.get(CoreAnnotations.SentenceIndexAnnotation.class);
            for (CoreLabel token : tokens) {
                System.out.println(sentInd + "_" + token.beginPosition() + "_" + token.word() + "_" + token.tag());
            }
        }
    }
}
```

## NER识别
```java
public class TestNer {
    public Annotation worker(Annotation annotation) throws Exception {
        new TestSent().worker(annotation);

        String basedir = "C:/Users/jian/Downloads/stanford-chinese-corenlp-2015-12-08-models/";
        Properties props = new Properties();
        props.setProperty("ner.model", basedir + "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz");
        props.setProperty("ner.applyNumericClassifiers", "false");
        props.setProperty("ner.useSUTime", "false");
        CRFClassifier<CoreLabel> classifier = new CRFClassifier<CoreLabel>(props);
        classifier.loadClassifierNoExceptions(basedir + "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz", props);

        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            List<CoreLabel> output = classifier.classifySentenceWithGlobalInformation(tokens, annotation, sentence);
            if (output != null) {
                for (int i = 0; i < tokens.size(); ++i) {
                    //String neTag = output.get(i).get(CoreAnnotations.NamedEntityTagAnnotation.class);
                    String neTag = output.get(i).get(CoreAnnotations.AnswerAnnotation.class);
                    tokens.get(i).setNER(neTag);
                }
            } else {
                for (CoreLabel token : tokens) {
                    token.setNER(classifier.backgroundSymbol());
                }
            }
        }
        return annotation;
    }

    public static void main(String[] args) throws Exception {
        String text = "何剑你好，听说你去中国人民大学参观了！四川省政府和腾讯成都分公司领导在青岛开会。";
        Annotation annotation = new Annotation(text);
        new TestNer().worker(annotation);
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            Integer sentInd = sentence.get(CoreAnnotations.SentenceIndexAnnotation.class);
            for (CoreLabel token : tokens) {
                System.out.println(sentInd + "_" + token.beginPosition() + "_" + token.word() + "_" + token.ner());
            }
        }
    }
}
```

参考pipeline的执行过程，以上`worker`方法还可以实现如下：
```java
    public Annotation newWorker(Annotation annotation) throws Exception {
        new TestSent().worker(annotation);

        String basedir = "C:/Users/jian/Downloads/stanford-chinese-corenlp-2015-12-08-models/";
        Properties props = new Properties();
        props.setProperty("ner.combinationMode", "normal");
        NERClassifierCombiner classifier = new NERClassifierCombiner(false, false, props, basedir + "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz");

        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            List<CoreLabel> output = classifier.classifySentenceWithGlobalInformation(tokens, annotation, sentence);
            if (output != null) {
                for (int i = 0; i < tokens.size(); ++i) {
                    String neTag = output.get(i).get(CoreAnnotations.NamedEntityTagAnnotation.class);
                    tokens.get(i).setNER(neTag);
                }
            } else {
                for (CoreLabel token : tokens) {
                    token.setNER(classifier.backgroundSymbol());
                }
            }
        }
        return annotation;
    }
```

使用`public NERClassifierCombiner(boolean applyNumericClassifiers, boolean useSUTime, Properties nscProps, String... loadPaths)`的好处是可以加载多个模型，还可使用`props.setProperty("ner.combinationMode", "normal")`控制模型结合模式。

## 使用pipeline
```java
public class TestPipe {
    public static void main(String[] args) throws Exception {
        String basedir = "C:/Users/jian/Downloads/stanford-chinese-corenlp-2015-12-08-models/";

        Properties props = new Properties();
        props.setProperty("annotators", "segment, ssplit, pos, ner");
        // segment
        props.setProperty("customAnnotatorClass.segment", "edu.stanford.nlp.pipeline.ChineseSegmenterAnnotator");
        props.setProperty("segment.NormalizationTable", basedir + "edu/stanford/nlp/models/segmenter/chinese/norm.simp.utf8");
        props.setProperty("segment.normTableEncoding", "UTF-8");
        props.setProperty("segment.model", basedir + "edu/stanford/nlp/models/segmenter/chinese/pku.gz");
        props.setProperty("segment.sighanCorporaDict", basedir + "edu/stanford/nlp/models/segmenter/chinese");
        props.setProperty("segment.serDictionary", basedir + "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz");
        props.setProperty("segment.sighanPostProcessing", "true");
        // sentence split
        props.setProperty("ssplit.boundaryTokenRegex", "[.]|[!?]+|[。]|[！？]+");
        // pos
        props.setProperty("pos.model", basedir + "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger");
        // ner
        props.setProperty("ner.model", basedir + "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz");
        props.setProperty("ner.applyNumericClassifiers", "false");
        props.setProperty("ner.useSUTime", "false");

        // pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Initialize an Annotation
        Annotation annotation = new Annotation("何剑你好，听说你去中国人民大学参观了！四川省政府和腾讯成都分公司领导在青岛开会。");

        // run all the selected Annotators on this text
        pipeline.annotate(annotation);

        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
            Integer sentInd = sentence.get(CoreAnnotations.SentenceIndexAnnotation.class);
            for (CoreLabel token : tokens) {
                System.out.println(sentInd + "_" + token.beginPosition() + "_" + token.word() + "_" + token.ner());
            }
        }
    }
}
```

如无特别需求，建议使用pipeline方式，不用太操心。

## 参考资料：
- [a suite of core NLP tools](http://stanfordnlp.github.io/CoreNLP/)
- [Chinese Natural Language Processing](http://nlp.stanford.edu/projects/chinese-nlp.shtml)
- [stanfordnlp/CoreNLP](https://github.com/stanfordnlp/CoreNLP)
- [Annotator dependencies](http://stanfordnlp.github.io/CoreNLP/dependencies.html)