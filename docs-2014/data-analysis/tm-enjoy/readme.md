title: 玩玩文本挖掘
date: 2015-12-18
tags: [TextMining,R]
---
文本挖掘的常见方法探索，主要包括词频分析、wordcloud展现、主题模型、文本分类、分类评价等。分类主要包括无监督分类：系统聚类、KMeans、string kernals，有监督分类：KNN、SVM。

<!--more-->
## 文本挖掘概念
将数据挖掘的成果用于分析以自然语言描述的文本，这种方法被称为文本挖掘(Text Mining)或文本知识发现(Knowledge Discovery in Text)。文本挖掘主要过程：特征抽取、特征选择、文本分类、文本聚类、模型评价。

主题模型是专门抽象一组文档所表达“主题”的统计技术。最早的模型是 Probabilistic Latent Semantic Indexing (PLSI)，后来 Latent Dirichlet Allocation (LDA，潜在狄利克雷分配模型)成为了最常见的主题模型，它可以认为是 PLSI 的泛化形式。LDA 主题模型涉及到贝叶斯理论、Dirichlet 分布、多项分布、图模型、变分推断、EM 算法、Gibbs 抽样等知识。

## 数据预处理
数据来源于[搜狗实验室](http://www.sogou.com/labs/resources.html)的`新闻数据 > 全网新闻数据`迷你版样例[数据网址](http://download.labs.sogou.com/dl/sogoulabdown/SogouCA/news_tensite_xml.smarty.zip)。下载完成后，把数据预处理为 txt 文件，要求每个新闻处理为一行：
```
txts <- read.table("news_tensite_xml.smarty.dat", stringsAsFactors=F, encoding="UTF-8")
txts <- apply(txts, 2, gsub, pattern="<.*?>", replacement="")
news <- data.frame(title=txts[seq(4,nrow(txts),by=6),1], content=txts[seq(5,nrow(txts),by=6),1])
write.table(news, file="news_tensite_xml.smarty.txt", row.names=F, col.names=F, quote=F, sep=",,,", fileEncoding="UTF-8")
```

## 读取语料库
```
# plan A
con0 <- file("news_tensite_xml.smarty.txt", "r", encoding="UTF-8")
txts <- readLines(con0)
close(con0)
# plan B
txts <- readLines("news_tensite_xml.smarty.txt", encoding="UTF-8")
library(Rwordseg)
txts <- sapply(txts, segmentCN, returnType="tm", USE.NAMES=F)
library(tm)
# 从字符向量创建语料库
dat_corpus <- Corpus(VectorSource(txts))
# 从特定目录创建语料库
# txts <- system.file("texts", "txt", package="tm")
# ovid <- Corpus(DirSource(txts), readerControl=list(language="lat"))
# 信息转化
dat_corpus <- tm_map(dat_corpus, tolower) #小写变化
dat_corpus <- tm_map(dat_corpus, removeWords, c("的")) #去除停用词
```

## wordcloud展示
```
tmp_tdm <- TermDocumentMatrix(dat_corpus, control=list(removePunctuation=T,wordLengths=c(2,Inf),weighting=weightTf))
library(wordcloud)
tmp_tdm <- as.matrix(tmp_tdm)
# doc 1 wordcloud
png("png_lab_wordcloud.png", width=800, height=800)
wordcloud(rownames(tmp_tdm), tmp_tdm[,1], scale=c(7,1), max.words=100, colors=rainbow(100))
title(main="lab_wordcloud")
dev.off()
```

## 主题模型分析
```
tmp_dtm <- DocumentTermMatrix(dat_corpus, control=list(removePunctuation=T,wordLengths=c(2,Inf),weighting=weightTf))
library(topicmodels)
# plan LDA
lda <- LDA(tmp_dtm, k=10, control=list(alpha=0.1))
# plan CTM
ctm <- CTM(tmp_dtm, k=10)
get_terms(lda, 5) #the most likely terms
get_topics(lda, 5) #the most likely topics
# Determine the posterior probabilities of the topics for each document
# and of the terms for each topic for a fitted topic model.
tmp_post <- posterior(lda, newdata=tmp_dtm)
str(tmp_post)
```

## 文本分类·无监督·kmeans
```
dat_topics <- as.data.frame(tmp_post$topics)
# kmeans
tmp_kms <- kmeans(dat_topics, 5)
library(clue)
# 计算最大共同分类率
cl_agreement(tmp_kms, as.cl_partition(dat_type), "diag")
```

String Kernel 是这样一种 Kernel 方法，它根据两个字符串的所有公共子串计算它们的相似度，String Kernel 目前最快的算法是基于 Suffix Tree 或 Suffix Array 的方法。

## 文本分类·有监督·SVM
把数据随机抽取 90% 作为学习集，剩下 10% 作为测试集。实际应用中应该进行交叉检验，这里简单起见，只进行一次抽取。
```
dat_topics <- as.data.frame(tmp_post$topics)
dat_topics$type <- rep(c("a","b"), length.out=nrow(dat_topics)) #造假
# create test and training set
dat_index <- sample.int(nrow(dat_topics), size=floor(nrow(dat_topics)*0.8))
dat_train <- dat_topics[dat_index,]
dat_test <- dat_topics[-dat_index,]
# train a support vector machine
m_filter <- ksvm(type~., data=dat_train, kernel="rbfdot", kpar=list(sigma=0.05), C=5, cross=3)
# predict type on the test set
m_predict <- predict(m_filter, dat_test[,1:10])
# Check results
table(m_predict, dat_test[,11])
```

[本文整理至网络](#)