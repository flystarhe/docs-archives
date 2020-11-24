title: Python | Magic Power
date: 2017-10-23
tags: [Python]
---
学习Python时维护一个经常使用的"窍门"列表,不论何时看到一段觉得"酷,这样也行!"的代码时.在一个例子中,在StackOverflow,在开源码软件中,等等.尝试理解它,然后把它添加到列表中.

<!--more-->
## 环境变量
使用环境变量完成配置:
```python
import os
# Get Environment variables
ENVIRON = os.environ.copy()
# Add Environment variable
os.environ['JAVA_HOME'] = 'home/hejian/apps/jdk1.8.0_161'
```

## 分拆
```python
>>> a, b, c = 1, 2, 3
>>> a, b, c = (i for i in range(3))
>>> a, b, c = [1, 2, 3]
>>> a, (b, c), d = [1, (2, 3), 4]
>>> a, *b, c = [1, 2, 3, 4, 5]
## a 1
## b [2, 3, 4]
## c 5
```

## 步长与负索引
```python
>>> a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> a[::2]
[0, 2, 4, 6, 8, 10]
>>> a[2:8:2]
[2, 4, 6]
>>> a[::-1]
## [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

## zip
```python
>>> a = [1, 2, 3]
>>> b = ['a', 'b', 'c']
>>> z = zip(a, b)
>>> z
[(1, 'a'), (2, 'b'), (3, 'c')]
>>> zip(*z)
[(1, 2, 3), ('a', 'b', 'c')]
```

合并相邻的列表项:
```python
>>> a = [1, 2, 3, 4, 5, 6]
>>> zip(*([iter(a)] * 2))
[(1, 2), (3, 4), (5, 6)]
>>> group_adjacent = lambda a, k: zip(*([iter(a)] * k))
>>> group_adjacent(a, 3)
[(1, 2, 3), (4, 5, 6)]
>>> group_adjacent = lambda a, k: zip(*(a[i::k] for i in range(k)))
>>> group_adjacent(a, 3)
[(1, 2, 3), (4, 5, 6)]
```

生成滑动窗口(ngram):
```python
>>> from itertools import islice
>>> def n_grams(a, n):
...     z = (islice(a, i, None) for i in range(n))
...     return zip(*z)
...
>>> a = [1, 2, 3, 4, 5, 6]
>>> n_grams(a, 3)
[(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
```

## 摊平列表
```python
>>> a = [[1, 2], [3, 4], [5, 6]]
>>> list(itertools.chain(a))
[1, 2, 3, 4, 5, 6]
>>> sum(a, [])
[1, 2, 3, 4, 5, 6]
>>> [x for l in a for x in l]
[1, 2, 3, 4, 5, 6]
```

## 命名序列
```python
>>> Point = collections.namedtuple('Point', ['x', 'y'])
>>> p = Point(x=1.0, y=2.0)
>>> p
Point(x=1.0, y=2.0)
>>> p.x
1.0
>>> p.y
2.0
```

## 多重集
最常见的元素:
```python
>>> A = collections.Counter([1, 1, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7])
>>> A
Counter({3: 4, 1: 2, 2: 2, 4: 1, 5: 1, 6: 1, 7: 1})
>>> A.most_common(1)
[(3, 4)]
>>> A.most_common(3)
[(3, 4), (1, 2), (2, 2)]
```

相关运算:
```python
>>> A = collections.Counter([1, 2, 2])
>>> B = collections.Counter([2, 2, 3])
>>> A
Counter({2: 2, 1: 1})
>>> B
Counter({2: 2, 3: 1})
>>> A | B
Counter({2: 2, 1: 1, 3: 1})
>>> A & B
Counter({2: 2})
>>> A + B
Counter({2: 4, 1: 1, 3: 1})
>>> A - B
Counter({1: 1})
```

## 缺省字典
```python
>>> m = collections.defaultdict(int)
>>> m['a']
0
>>> m = collections.defaultdict(str)
>>> m['a']
''
>>> m = collections.defaultdict(lambda: '[default value]')
>>> m['a']
'[default value]'
```

用缺省字典表示简单的树:
```python
>>> import json
>>> tree = lambda: collections.defaultdict(tree)
>>> root = tree()
>>> root['menu']['id'] = 'file'
>>> root['menu']['value'] = 'File'
>>> root['menu']['menuitems']['new']['value'] = 'New'
>>> root['menu']['menuitems']['new']['onclick'] = 'new();'
>>> root['menu']['menuitems']['open']['value'] = 'Open'
>>> root['menu']['menuitems']['open']['onclick'] = 'open();'
>>> root['menu']['menuitems']['close']['value'] = 'Close'
>>> root['menu']['menuitems']['close']['onclick'] = 'close();'
>>> print json.dumps(root, sort_keys=True, indent=4, separators=(',', ': '))
{
    "menu": {
        "id": "file",
        "menuitems": {
            "close": {
                "onclick": "close();",
                "value": "Close"
            },
            "new": {
                "onclick": "new();",
                "value": "New"
            },
            "open": {
                "onclick": "open();",
                "value": "Open"
            }
        },
        "value": "File"
    }
}
```

## 最大最小
```python
>>> a = [random.randint(0, 100) for _ in range(100)]
>>> heapq.nsmallest(5, a)
[3, 3, 5, 6, 8]
>>> heapq.nlargest(5, a)
[100, 100, 99, 98, 98]
```

## 参考资料:
- [你可能不知道的30个Python语言的特点技巧](http://sahandsaba.com/thirty-python-language-features-and-tricks-you-may-not-know.html)