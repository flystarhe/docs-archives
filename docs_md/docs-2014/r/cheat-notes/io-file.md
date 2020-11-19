title: R入门小抄 | 文件系统
date: 2015-05-20
tags: [R]
---
目录与文件操作.

<!--more-->
## 查看目录
```
getwd() #当前的目录
list.dirs() #查看当前目录的子目录
dir() #查看当前目录的子目录和文件 --list.files()
dir(path="/home/root/R") #查看指定目录的子目录和文件
dir(path="/home/root/R", pattern='^R') #只列出以字母R开头的子目录或文件
```

## 创建目录
```
dir.create("create") #在当前目录下新建一个目录
dir.create(path="p1/p2/p3", recursive=TRUE) #递归创建
system("tree") #通过系统命令查看目录结构
```

## 检查目录
```
temp = dir(full.names=TRUE)
file.exists("create") #目录是否存在
file.access(c(temp,"notExists"), 0) == 0 #检查文件或目录是否存在
file.access(c(temp,"notExists"), 1) == 0 #检查文件或目录可是执行
file.access(c(temp,"notExists"), 2) == 0 #检查文件或目录是否可写
file.access(c(temp,"notExists"), 4) == 0 #检查文件或目录是否可读
```

## 其他函数
```
file.rename("create", "temp") #目录重命名
unlink("temp", recursive=TRUE) #删除目录
file.path("p1", "p2", "p3") #拼接目录字符串 --p1/p2/p3
dirname("/home/root/R/fs/readme.txt") #最底层子目录 --/home/root/R/fs
basename("/home/root/R/fs") #最底层子目录或文件名 --fs
basename("/home/root/R/fs/readme.txt") #最底层子目录或文件名 --readme.txt
```

## 查看文件
```
file.create("A.txt") #创建一个空文件
cat("file B\n", file="B.txt") #创建一个有内容的文件
file.exists("A.txt") #文件是否存在
file_test("-d", "A.txt") #判断是否是目录
file_test("-f", "A.txt") #判断是否是文件
```

## 其他函数
```
readLines("B.txt") #读文件
file.rename("A.txt", "AA.txt") #文件重命名
file.remove("AA.txt", "B.txt", "readme.txt") #删除文件
unlink("AA.txt") #删除文件
```

## .RData
```r
ls()
## [1] "x" "y" "z"
## save data
save(list=c("x","y","z"),file="_save_test.RData")
## load data
rm(list=ls(all=TRUE))
load("_save_test.RData",.GlobalEnv)
## save data//all
save(list=ls(all=TRUE),file="_all.RData")
## load data//all
load("_all.RData",.GlobalEnv)
```

## .csv
```r
write.csv(iris,file="test.csv",fileEncoding="utf-8",row.names=FALSE)
read.csv("test.csv",sep=",",fileEncoding="utf-8",stringsAsFactors=TRUE)
```

## .arff
```r
library(RWeka)
## Prepare data
write.arff(iris,"iris.arff",eol="\n")
## Use some example data.
dat_iris <- read.arff("iris.arff")
## Identify a decision tree.
m <- J48(play~., data = w)
## Use 10 fold cross-validation.
e <- evaluate_Weka_classifier(m,
cost = matrix(c(0,2,1,0), ncol = 2),
numFolds = 10, complexity = TRUE,
seed = 123, class = TRUE)
summary(e)
e$details
```
