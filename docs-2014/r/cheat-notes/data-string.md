title: R入门小抄 | 字符串处理
date: 2015-05-20
tags: [R]
---
字符串处理.

<!--more-->
```
x = 'hello\rwold\n'
```

## 字符串输出
```
cat(x)
## woldo --遇到\r光标移到头接着打印wold覆盖了之前的hell变成woldo
print(x)
## [1] "hello\rwold\n"
```

## 字符串长度
```
nchar(x)
## 11 --字符串长度
length(x)
## 1  --向量中元素的个数
```

## 字符串拼接
```
board = paste('b', 1:4, sep='-')
## [1] "b-1" "b-2" "b-3" "b-4"
mm = paste('mm', 1:3, sep='-')
## [1] "mm-1" "mm-2" "mm-3"
outer(board, mm, paste, sep=':')
##      [,1]       [,2]       [,3]      
## [1,] "b-1:mm-1" "b-1:mm-2" "b-1:mm-3"
## [2,] "b-2:mm-1" "b-2:mm-2" "b-2:mm-3"
## [3,] "b-3:mm-1" "b-3:mm-2" "b-3:mm-3"
## [4,] "b-4:mm-1" "b-4:mm-2" "b-4:mm-3"
```

## 拆分提取
```
board
## [1] "b-1" "b-2" "b-3" "b-4"
substr(board, 3, 3)
## [1] "1" "2" "3" "4"
strsplit(board, '-', fixed=T)
## [[1]]
## [1] "b" "1"
## 
## [[2]]
## [1] "b" "2"
## 
## [[3]]
## [1] "b" "3"
## 
## [[4]]
## [1] "b" "4"
```

## 修改
```
board
## [1] "b-1" "b-2" "b-3" "b-4"
sub('-', '.', board, fixed=T)
## [1] "b.1" "b.2" "b.3" "b.4" --修改指定字符
mm
## [1] "mm-1" "mm-2" "mm-3"
sub('m', 'p', mm)
## [1] "pm-1" "pm-2" "pm-3" --替换首个匹配项
gsub('m','p',mm)
## [1] "pp-1" "pp-2" "pp-3" --替换全部匹配项
```

## 查找
```
mm = c(mm, 'mm4')
## [1] "mm-1" "mm-2" "mm-3" "mm4" 
grep('-', mm)
## [1] 1 2 3 --索引
regexpr('-', mm)
## [1]  3  3  3 -1 --匹配成功会返回位置信息，没有找到则返回-1
## attr(,"match.length")
## [1]  1  1  1 -1
## attr(,"useBytes")
## [1] TRUE
attr(regexpr('-', mm), "match.length")
## [1]  1  1  1 -1
attr(regexpr('-', mm), "useBytes")
## [1] TRUE
```

## str2num
```r
> dat = c('a','b','a','c','b','c')
> dat
[1] "a" "b" "a" "c" "b" "c"
> dat1 = as.factor(dat)
> dat1
[1] a b a c b c
Levels: a b c
> levels(dat1)
[1] "a" "b" "c"
> as.numeric(dat1)
[1] 1 2 1 3 2 3
> levels(dat1)[as.numeric(dat1)] == dat
[1] TRUE TRUE TRUE TRUE TRUE TRUE
> all(levels(dat1)[as.numeric(dat1)] == dat)
[1] TRUE
> levels(dat1)[dat1] == dat
[1] TRUE TRUE TRUE TRUE TRUE TRUE
> all(levels(dat1)[dat1] == dat)
[1] TRUE
> mode(dat1)
[1] "numeric"
```
