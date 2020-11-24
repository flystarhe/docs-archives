title: Python标准库itertools
date: 2017-10-12
tags: [Python]
---
标准库中的`itertools`包提供了灵活的生成循环器的工具.这些工具的输入大都是已有的循环器.这些工具完全可以自行使用Python实现,该包只是提供了一种比较标准,高效的实现方式.

<!--more-->
## 无穷循环器
```python
import itertools
itertools.count(5, 2)     #从5开始的整数循环器,每次增加2,即(5,7,9 ..)
itertools.cycle('abc')    #重复序列的元素,即(a,b,c,a,b,c ..)
itertools.repeat(1.2)     #重复1.2,构成无穷循环器,即(1.2,1.2 ..)
itertools.repeat(10, 5)   #重复10,共5次
```

## 函数式工具
函数式编程是将函数本身作为处理对象的编程范式.在Python中,函数也是对象,因此可以轻松的进行一些函数式的处理,比如`map()`,`filter()`,`reduce()`.`itertools`包含类似的工具,这些函数接收函数作为参数,并将结果返回为一个循环器:
```python
from itertools import *
rlt = imap(pow, [1, 2, 3], [1, 2, 3])
for num in rlt:
    print(num)
```

`imap()`与`map()`功能相似,只不过返回的不是序列,而是一个循环器.`pow()`将依次作用于后面两个列表的每个元素,并收集函数结果,组成返回的循环器.还可以用下面的函数,`pow()`将依次作用于每个`tuple`:
```python
starmap(pow, [(1, 1), (2, 2), (3, 3)])
```

`ifilter()`与`filter()`类似,如果函数返回True,则收集元素.只是返回的是一个循环器:
```python
ifilter(lambda x: x > 5, [2, 3, 5, 6, 7])
```

与上面类似,但收集返回False的元素:
```python
ifilterfalse(lambda x: x > 5, [2, 3, 5, 6, 7])
```

当函数返回True时,收集元素.一旦函数返回False,则停止:
```python
takewhile(lambda x: x < 5, [1, 3, 6, 7, 1])
```

当函数返回True时,跳过元素.一旦函数返回False,则开始收集剩下的所有元素:
```python
dropwhile(lambda x: x < 5, [1, 3, 6, 7, 1])
```

## 组合工具
通过组合原有循环器,来获得新的循环器:
```python
chain([1, 2, 3], [4, 5, 7])      #连接两个循环器成为一个
```

多个循环器集合的笛卡尔积,相当于嵌套循环:
```python
product('abc', [1, 2])
```

排列组合:
```python
permutations('abc', 2)
combinations('abc', 2)
combinations_with_replacement('abc', 2)
#与上面类似,但允许两次选出的元素重复,即多了(aa,bb,cc)
```

## groupby
将key函数作用于原循环器的各个元素。根据key函数结果，将拥有相同函数结果的元素分到一个新的循环器。每个新的循环器以函数返回结果为标签。
```python
def height_class(h):
    if h > 180:
        return "tall"
    elif h < 160:
        return "short"
    else:
        return "middle"

heights = [191, 158, 159, 165, 170, 177, 181, 182, 190]

heights = sorted(heights, key=height_class)
for m, n in groupby(heights, key=height_class):
    print(m)
    print(list(n))
```

>注意,`groupby()`之前需要使用`sorted()`对原循环器的元素,根据`key`进行排序,让同组元素先在位置上靠拢.

## 参考资料:
- [docs/itertools](https://docs.python.org/3.6/library/itertools.html)