title: Python入门指南
date: 2016-10-14
tags: [Python]
---
Python是一门简单易学且功能强大的编程语言。它拥有高效的高级数据结构，并且能够用简单而又高效的方式进行面向对象编程。优雅的语法和动态类型，再结合它的解释性，使其在大多数平台的许多领域成为编写脚本或开发应用程序的理想语言。

<!--more-->
## requirements.txt
优雅的Python项目,会包含一个`requirements.txt`文件,用于记录所有依赖包及其精确的版本号,以便新环境部署.

生成:
```
pip freeze > requirements.txt
```

使用:
```
pip install -r requirements.txt
conda install -y --file requirements.txt
# This file may be used to create an environment using:
conda create -y --name <env> --file <this file>
```

## 变量是否存在
常用的两个方法是`dir()`和`locals()`,相对来说`locals()`更快:
```python
'pool' in dir()
'pool' not in dir()
'pool' in locals()
'pool' not in locals()
```

`dir()`,如果没有参数调用,则返回当前范围中的名称.`locals`,更新并以字典形式返回当前局部符号表.`globals()`,返回当前全局符号表.通过`sys.modules`可以查看所有的已加载并且成功的模块.

## Python并发编程
```python
import multiprocessing
import time

def test(tries):
    time.sleep(1)
    print('do: %d' % tries)
    return 'do: %d' % tries
```

### 方案1
简单的非阻塞,异步:
```python
# case 1
pool = multiprocessing.Pool(processes=2)
for i in range(3):
    print('add %d' % i)
    res = pool.apply_async(test, (i,))  # 非阻塞,异步
print('wait~ wait~ wait')
pool.close()  # 不再接受新任务
pool.join()  # 等待子进程结束
print('end~ end~ end')
```

输出为:

    add 0
    add 1
    add 2
    wait~ wait~ wait
    do: 0
    do: 1
    do: 2
    end~ end~ end

### 方案2
简单的阻塞,非异步:
```python
# case 2
pool = multiprocessing.Pool(processes=2)
for i in range(3):
    print('add %d' % i)
    res = pool.apply(test, (i,))  # 阻塞
print('wait~ wait~ wait')
pool.close()  # 不再接受新任务
pool.join()  # 等待子进程结束
print('end~ end~ end')
```

输出为:

    add 0
    do: 0
    add 1
    do: 1
    add 2
    do: 2
    wait~ wait~ wait
    end~ end~ end

### 方案3
异步,需要收集结果:
```python
# case 3
res = []
pool = multiprocessing.Pool(processes=2)
for i in range(3):
    res.append(pool.apply_async(test, (i,)))
print('wait~ wait~ wait')
pool.close()  # 不再接受新任务
pool.join()  # 等待子进程结束
print('end~ end~ end')
for i in res:
    print('res: ', i.get())
```

输出为:

    wait~ wait~ wait
    do: 0
    do: 1
    do: 2
    end~ end~ end
    res:  do: 0
    res:  do: 1
    res:  do: 2

### 方案4
更快捷的方式,使用`Pool.map`:
```python
# case 4
ts = time.time()
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
result = pool.map(test, range(3))
pool.close()
print('time: %s' % (time.time() - ts))
print(result)
```

输出为:

    time: 1.0127575397491455
    ['do: 0', 'do: 1', 'do: 2']

### 方案5
共享变量:
```python
from multiprocessing import Pool, Value
import os
index = Value('i', 0)
def test(x):
    with index.get_lock():
        tmp = index.value
        index.value += 1
    return (x, tmp, 'pid: %d' % os.getpid(), 'parent.pid: %d' % os.getppid())

res = []
pool = Pool(processes=2)
for i in range(10):
    res.append(pool.apply_async(test, (i,)))
pool.close()
pool.join()
for i in res:
    print(i.get())
```

输出为:

    (0, 0, 'pid: 13796', 'parent.pid: 13793')
    (1, 1, 'pid: 13797', 'parent.pid: 13793')
    (2, 2, 'pid: 13796', 'parent.pid: 13793')
    (3, 3, 'pid: 13797', 'parent.pid: 13793')
    (4, 4, 'pid: 13796', 'parent.pid: 13793')
    (5, 5, 'pid: 13797', 'parent.pid: 13793')
    (6, 6, 'pid: 13797', 'parent.pid: 13793')
    (7, 7, 'pid: 13796', 'parent.pid: 13793')
    (8, 8, 'pid: 13797', 'parent.pid: 13793')
    (9, 9, 'pid: 13797', 'parent.pid: 13793')

## Python获取当前目录
`sys.path`是Python会去寻找模块的搜索路径列表,`sys.path[0]`和`sys.argv[0]`是一回事因为Python会自动把`sys.argv[0]`加入`sys.path`.

如果你在`/lab/test/`目录下执行`python getpath/getpath.py`,那么`os.getcwd()`会输出`/lab/test/`,`sys.path[0]`会输出`/lab/test/getpath/`.

## None
None是一个特殊的常量。None不是0，None不是空字符串，None和任何其他的数据类型比较永远返回False。None有自己的数据类型NoneType。你可以将None复制给任何变量，但是你不能创建其他NoneType对象。

## 条件控制
```python
age = 3
if age < 0:
    print("你是在逗我吧！")
elif age == 1:
    print("相当于 14 岁的人。")
elif age == 2:
    print("相当于 22 岁的人。")
else:
    print("对应人类年龄：", 22 + (age-2)*5)
```

## 循环语句
while循环：
```python
n, sum, counter = 10, 0, 1
while counter <= n:
    sum += counter
    counter += 1
print("1 到 {0} 之和为: {1}".format(n, sum))
```

while循环使用else语句，在条件语句为false时执行else的语句块：
```python
count = 0
while count < 5:
    print(count, "小于 5")
    count = count + 1
else:
    print(count, "大于或等于 5")
```

for循环可以遍历任何序列的项目，如一个列表或者一个字符串：
```python
languages = ["C", "C++", "Perl", "Python"]
for x in languages:
    print(x)
else:
    print("没有循环数据！")
```

continue语句用于跳过当次循环，break语句用于跳出当前循环体：
```python
sites = ["Baidu", "Google","Runoob","Taobao"]
for site in sites:
    if site == "Runoob": break
    print("循环数据：" + site)
```

`range()`函数能生成数字数列：
```python
for i in range(5):  # 0 1 2 3 4
    print(i)
```

## 迭代器与生成器
迭代是Python最强大的功能之一，是访问集合元素的一种方式。迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。迭代器有两个基本的方法：`iter()`和`next()`。字符串，列表或元组对象都可用于创建迭代器：
```python
>>> list = [1, 2, 3, 4]
>>> it = iter(list)    # 创建迭代器对象
>>> print(next(it))   # 输出迭代器的下一个元素
1
>>> it = iter(list)    # 创建迭代器对象
>>> for x in it:
...     print(x, end=" ")
1 2 3 4
```

使用了`yield`的函数被称为生成器`generator`。跟普通函数不同的是，生成器是一个返回迭代器的函数，只能用于迭代操作。在调用生成器运行的过程中，每次遇到`yield`时函数会暂停并保存当前所有的运行信息，返回`yield`的值。并在下一次执行`next()`方法时从当前位置继续运行。以下实例使用`yield`实现斐波那契数列：
```python
import sys
def fibonacci(n): # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n): 
            return
        yield a
        a, b = b, a + b
        counter += 1

f = fibonacci(10) # f 是一个迭代器
while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()
```

## 遍历技巧
在字典中遍历时，关键字和对应的值可以使用`items()`方法同时解读出来：
```python
knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
    print(k, v)
# gallahad the pure
# robin the brave
```

在序列中遍历时，索引位置和对应值可以使用`enumerate()`函数同时得到：
```python
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i, v)
# 0 tic
# 1 tac
# 2 toe
```

同时遍历两个或更多的序列，可以使用`zip()`组合：
```python
>>> questions = ['name', 'quest', 'favorite color']
>>> answers = ['lancelot', 'the holy grail', 'blue']
>>> for q, a in zip(questions, answers):
...     print('What is your {0}?  It is {1}.'.format(q, a))
```

要反向遍历一个序列，首先指定这个序列，然后调用`reversesd()`函数：
```python
>>> for i in reversed(range(1, 10, 2)):
...     print(i)
```

要按顺序遍历一个序列，使用`sorted()`函数返回一个已排序的序列，并不修改原值：
```python
>>> basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
>>> for f in sorted(set(basket)):
...     print(f)
```

## 函数
Python定义函数使用`def`关键字，一般格式如下：

    def 函数名(参数列表):
        函数体

>Python中，所有参数（变量）都是按引用传递。如果你在函数里修改了参数，那么在调用这个函数的函数里，原始的参数也被改变了。

你可能需要一个函数能处理比当初声明时更多的参数。加了`*`的变量名会存放所有未命名的变量参数。如果在函数调用时没有指定参数，它就是一个空元组。如下实例：
```python
# 可写函数说明
def printinfo(arg1, *vartuple):
    print("输出：")
    print(arg1)
    for var in vartuple:
        print(var)
    return;
# 调用printinfo 函数
printinfo(10);
printinfo(70, 60, 50);
```

Python使用`lambda`来创建匿名函数。`lambda`函数的语法只包含一个语句，如下：
```python
# lambda [arg1[, arg2 ..]]: expression
sum = lambda arg1, arg2: arg1 + arg2;  # 匿名函数
print("相加后的值为：", sum(10, 20))  # 调用sum函数
```

## 类
一个简单的类实例：
```python
class MyClass:
    """一个简单的类实例"""
    i = 12345
    def f(self):
        return 'hello world'
```

类可能会定义一个名为`__init__()`的特殊方法（构造方法），类的实例化操作会自动调用`__init__()`方法：
```python
def __init__(self):
    self.data = []
def __init__(self, realpart, imagpart):
    self.r = realpart
    self.i = imagpart
```

类的方法与一般函数定义不同，类方法必须包含参数`self`，且为第一个参数：
```python
# 类定义
class people:
    # 定义基本属性
    name = ''
    age = 0
    # 定义私有属性
    __weight = 0
    # 定义构造方法
    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w
    # 类的普通方法
    def speak(self):
        print("{0} 说：我 {1} 岁了".format(self.name, self.age))
# 实例化类
p = people('runoob', 10, 30)
```

Python同样支持类的继承，派生类的定义如下所示：
```python
class student(people):
    grade = ''
    def __init__(self, n, a, w, g):
        # 调用父类的构函
        people.__init__(self, n, a, w)
        self.grade = g
    # 覆写父类的方法
    def speak(self):
        print("{0} 说：我 {1} 岁了，我在读 {2} 年级".format(self.name, self.age, self.grade))
```

类的专有方法：
```python
__init__    # 构造函数，在生成对象时调用
__del__     # 析构函数，释放对象时使用
__repr__    # 打印，转换
__setitem__ # 按照索引赋值
__getitem__ # 按照索引获取值，如 x[key]
__len__     # 获得长度
__cmp__     # 比较运算
__call__    # 函数调用
__add__     # 加运算
__sub__     # 减运算
__mul__     # 乘运算
__div__     # 除运算
__mod__     # 求余运算
__pow__     # 称方
```

## 正则表达式
`re.match`尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功就返回`None`：
```python
import re
# re.match(pattern, string, flags=0)
line = "Cats are smarter than dogs"
matchObj = re.match(r'(.*) are (.*?) .*', line, re.M|re.I)
if matchObj:
    print("matchObj.span() : ", matchObj.span())
    print("matchObj.group() : ", matchObj.group())
    print("matchObj.group(1) : ", matchObj.group(1))
    print("matchObj.group(2) : ", matchObj.group(2))
else:
    print("No match!!")
```

以上实例执行结果如下：

    matchObj.span() :  (0, 26)
    matchObj.group() :  Cats are smarter than dogs
    matchObj.group(1) :  Cats
    matchObj.group(2) :  smarter

`re.search`扫描整个字符串并返回第一个成功的匹配，匹配不成功则返回`None`：
```python
import re
# re.search(pattern, string, flags=0)
line = "Cats are smarter than dogs"
searchObj = re.search( r'(.*) are (.*?) .*', line, re.M|re.I)
if searchObj:
    print("searchObj.span() : ", searchObj.span())
    print("searchObj.group() : ", searchObj.group())
    print("searchObj.group(1) : ", searchObj.group(1))
    print("searchObj.group(2) : ", searchObj.group(2))
else:
    print("Nothing found!!")
```

以上实例执行结果如下：

    searchObj.span() :  (0, 26)
    searchObj.group() :  Cats are smarter than dogs
    searchObj.group(1) :  Cats
    searchObj.group(2) :  smarter

>`re.match`只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回`None`；而`re.search`匹配整个字符串，直到找到一个匹配。

Python的`re`模块提供了`re.sub`用于替换字符串中的匹配项：
```python
import re
# re.sub(pattern, repl, string, count=0)
phone = "2004-959-559 # 这是一个电话号码"
num = re.sub(r'#.*$', '', phone)  # 删除注释
print ("电话号码 : ", num)
num = re.sub(r'(\D+)', r'{\1}', phone)  # 标记非数字的内容
print ("电话号码 : ", num)
```

以上实例执行结果如下：

    电话号码 :  2004-959-559 
    电话号码 :  2004{-}959{-}559{ # 这是一个电话号码}

正则表达式可以包含一些可选标志修饰符来控制匹配的模式。修饰符被指定为一个可选的标志。多个标志可以通过按位`|`它们来指定。如`re.I|re.M`被设置成`I`和`M`标志：
```python
re.I    # 使匹配对大小写不敏感
re.L    # 做本地化识别（locale-aware）匹配
re.M    # 多行匹配，影响 ^ 和 $
re.S    # 使 . 匹配包括换行在内的所有字符
re.U    # 根据Unicode字符集解析字符。这个标志影响 \w, \W, \b, \B.
re.X    # 该标志通过给予你更灵活的格式以便你将正则表达式写得更易于理解。
```

## 参考资料：
- [Python 3 教程](http://www.runoob.com/python3/python3-tutorial.html)
- [Python Doc 3.6.0](http://www.pythondoc.com/pythontutorial3/index.html)