title: 各种相似性度量(Python)
date: 2017-07-14
tags: [相似性,距离]
---
距离度量的种类:欧氏距离,曼哈顿距离,切比雪夫距离,马氏距离,编辑距离,余弦距离,Ngram距离.及Python实现.

<!--more-->
## 欧氏距离
```python
import numpy as np
import math
def Euclidean(vec1, vec2):
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((np_vec1 - np_vec2)**2).sum())
```

## 曼哈顿距离
```python
import numpy as np
import math
def Manhattan(vec1, vec2):
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    return np.abs(np_vec1 - np_vec2).sum()
```

## 切比雪夫距离
```python
import numpy as np
import math
def Chebyshev(vec1, vec2):
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    return max(np.abs(np_vec1 - np_vec2))
```

## 马氏距离
```python
import numpy as np
import math
def Mahalanobis(vec1, vec2):
    np_vec1, np_vec2 = np.array(vec1), np.array(vec2)
    np_vec = np.array([np_vec1, np_vec2])
    sub = np_vec.T[0] - np_vec.T[1]
    inv_sub = np.linalg.inv(np.cov(np_vec1, np_vec2))
    return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))
```

## 编辑距离
[编辑距离](#),俄罗斯科学家Vladimir Levenshtein在1965年提出这个概念.又称Levenshtein距离,是指两个字串之间,由一个转成另一个所需的最少编辑操作次数.许可的编辑操作包括将一个字符替换成另一个字符,插入一个字符,删除一个字符.str1或str2的长度为0返回另一个字符串的长度.例如将`kitten`转成`sitting`:
```
kitten =(k to s)=> sitten =(e to i)=> sittin =( add g)=> sitting
```

字符串:
```python
import Levenshtein
def Edit_distance_str_1(str1, str2):
    num_dist = Levenshtein.distance(str1, str2)
    num_sim = 1 - num_dist / max(len(str1), len(str2))
    return {'dist': num_dist, 'sim': num_sim}

import editdistance
def Edit_distance_str_2(str1, str2):
    num_dist = editdistance.eval(str1, str2)
    num_sim = 1 - num_dist / max(len(str1), len(str2))
    return {'dist': num_dist, 'sim': num_sim}
```

序列:
```python
arr1 = ['abc', 'bc']
arr2 = ['abc', 'bd']

import editdistance
def Edit_distance_arr_1(arr1, arr2):
    num_dist = editdistance.eval(arr1, arr2)
    num_sim = 1 - num_dist / max(len(str1), len(str2))
    return {'dist': num_dist, 'sim': num_sim}

def Edit_distance_arr_2(arr1, arr2):
    size1 = len(arr1) + 1
    size2 = len(arr2) + 1
    temp = [0] * (size1 * size2)
    for i in range(size1):
        temp[i] = i
    for j in range(size2):
        temp[j*size1] = j
    for i in range(1, size1):
        for j in range(1, size2):
            fit2 = j * size1
            fit1 = fit2 - size1
            val1 = min(temp[fit2 + i -1], temp[fit1 + i]) + 1
            val2 = temp[fit1 + i - 1] + int(arr1[i-1]!=arr2[j-1])
            temp[fit2 + i] = min(val1, val2)
    num_dist = temp[-1]
    num_sim = 1 - num_dist / max(len(arr1), len(arr2))
    return {'dist': num_dist, 'sim': num_sim}
```

应用:DNA分析,拼字检查,语音辨识,抄袭侦测.

### 性能
```python
a = 'fsffvfdsbbdfvvdavavavavavava'
b = 'fvdaabavvvvvadvdvavavadfsfsdafvvav'

import difflib
%timeit difflib.SequenceMatcher(None, a, b).ratio()

import editdistance
%timeit (1 - editdistance.eval(a, b) / max(len(a), len(b)))
```

结论:
```
98.8 µs ± 274 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
1.47 µs ± 6.46 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

## Ngram距离
Ngram代表在给定序列中产生连续的N项,当序列句子时,每项就是单词.假设有一个字符串S,那么该字符串的Ngram就表示按长度N切分原词得到的词段,也就是S中所有长度为N的子字符串.设想如果有两个字符串,然后分别求它们的Ngram,那么就可以从它们的共有子串的数量这个角度去定义两个字符串间的[Ngram距离](#).以`N=2`为例对字符串`Gorbachev`和`Gorbechyov`进行分段,可得如下结果:
```
Go or rb ba ac ch he ev
Go or rb be ec ch hy yo ov
```

即可算得两个字符串之间的距离是`8 + 9 − 2 × 4 = 9`.代码:
```python
def Ngram_distance(str1, str2, n=2):
    tmp = ' ' * (n-1)
    str1 = tmp + str1 + tmp
    str2 = tmp + str2 + tmp
    set1 = set([str1[i:i+n] for i in range(len(str1)-(n-1))])
    set2 = set([str2[i:i+n] for i in range(len(str2)-(n-1))])
    setx = set1 & set2
    len1 = len(set1)
    len2 = len(set2)
    lenx = len(setx)
    num_dist = len1 + len2 - 2*lenx
    num_sim = 1 - num_dist / (len1 + len2)
    return {'dist': num_dist, 'sim': num_sim}
```

Ngram的推广,skip-Ngram产生的N项子序列中,各个项在原序列中不连续,而是跳了k个字.例如,对于句子:
```
the rain in Spain
```
其`2-grams`为子序列集合:
```
the rain，rain in，in Spain
```
其`1-skip-2-grams`为子序列集合:
```
the in, rain Spain
```
