title: Python数据类型
date: 2016-10-13
tags: [Python,数据类型]
---
Python3中有六个标准的数据类型：Number（数字），String（字符串），List（列表），Tuple（元组），Sets（集合），Dictionary（字典）。

<!--more-->
## Number（数字）
Python3支持int、float、bool、complex（复数）。在Python3里，只有一种整数类型int，表示为长整型，没有python2中的Long。像大多数语言一样，数值类型的赋值和计算都是很直观的。内置的`type()`函数可以用来查询变量所指的对象类型。
```python
>>> a, b, c, d = 20, 5.5, True, 4+3j
>>> print(type(a), type(b), type(c), type(d))
<class 'int'> <class 'float'> <class 'bool'> <class 'complex'>
>>> 5 + 4   # 加法
9
>>> 4.3 - 2 # 减法
2.3
>>> 3 * 7  # 乘法
21
>>> 2 / 4  # 除法，得到一个浮点数
0.5
>>> 2 // 4 # 除法，得到一个整数
0
>>> 17 % 3 # 取余 
2
>>> 2 ** 5 # 乘方
32
```

当你指定一个值时，Number对象就会被创建。您也可以使用del语句删除一些对象引用：
```python
var1 = 1
var2 = 10
del var1, var2
```

## String（字符串）
Python中的字符串用`'`或`"`括起来，同时使用反斜杠`\`转义特殊字符。字符串索引值以0为开始值，-1为从末尾的开始位置。`+`是字符串的连接符，`*`表示复制当前字符串，紧跟的数字为复制的次数。实例如下：
```python
str = 'Runoob'
print(str)          # 输出字符串
print(str[0:-1])    # 输出第一个个到倒数第二个的所有字符
print(str[0])       # 输出字符串第一个字符
print(str[2:5])     # 输出从第三个开始到第五个的字符
print(str[2:])      # 输出从第三个开始的后的所有字符
print(str * 2)      # 输出字符串两次
print(str + "TEST") # 连接字符串
```

以上实例输出结果如下：
```python
Runoob
Runoo
R
noo
noob
RunoobRunoob
RunoobTEST
```

>Python使用反斜杠`\`转义特殊字符，如果不想让反斜杠发生转义，可以在字符串前面添加一个`r`，表示原始字符串。另外，反斜杠`\`可以作为续行符，表示下一行是上一行的延续。也可以使用`"""`或`'''`跨越多行。

字符串格式化：
```python
>>> # 使用索引
>>> print('{1} 和 {0}'.format('Google', 'Runoob'))
Runoob 和 Google
>>> # 使用关键字参数
>>> print('{name}网址：{site}'.format(name='菜鸟教程', site='www.runoob.com'))
菜鸟教程网址：www.runoob.com
>>> # 保留到小数点后三位
>>> print('常量 PI 的值近似为 {0:.3f}。'.format(math.pi))
常量 PI 的值近似为 3.142。
>>> # 在 ':' 后传入一个整数, 可以保证该域至少有这么多的宽度
>>> print('{0:10} ==> {1:10d}'.format('name', 7))
name       ==>          7
>>> # 传入一个字典，然后使用方括号 '[]' 来访问键值
>>> print('Runoob: {0[Runoob]:d}; Google: {0[Google]:d}; Taobao: {0[Taobao]:d}'.format({'Google': 1, 'Runoob': 2, 'Taobao': 3}))
```

## List(列表)
List（列表）是Python中使用最频繁的数据类型。列表可以完成大多数集合类的数据结构实现。列表中元素的类型可以不相同，它支持数字，字符串甚至可以包含列表（所谓嵌套）。列表是写在`[]`之间，用逗号分隔开的元素列表。

和字符串一样，列表同样可以被索引和截取，索引值以0为开始值，-1为从末尾的开始位置，列表被截取后返回一个包含所需元素的新列表。`+`是列表连接运算符，`*`是重复操作。如下实例：
```python
list = ['abcd', 786 , 2.23, 'runoob', 70.2]
tinylist = [123, 'runoob']

print(list)            # 输出完整列表
print(list[0])         # 输出列表第一个元素
print(list[1:3])       # 从第二个开始输出到第三个元素
print(list[2:])        # 输出从第三个元素开始的所有元素
print(tinylist * 2)    # 输出两次列表
print(list + tinylist) # 连接列表
```

以上实例输出结果如下：
```python
['abcd', 786, 2.23, 'runoob', 70.2]
abcd
[786, 2.23]
[2.23, 'runoob', 70.2]
[123, 'runoob', 123, 'runoob']
['abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob']
```

与Python字符串不一样的是，列表中的元素是可以改变的：
```python
>>> a = [1, 2, 3, 4, 5, 6]
>>> a[0] = 9
>>> a[2:5] = [13, 14, 15]
>>> a
[9, 2, 13, 14, 15, 6]
>>> a[2:5] = []   # 删除
>>> a
[9, 2, 6]
>>> vec = [2, 4, 6]
>>> [3*x for x in vec]  # 列表推导式
[6, 12, 18]
>>> [3*x for x in vec if x > 3]  # 用 if 子句作为过滤器
[12, 18]
```

另外，列表类型也有一些内置的函数，例如`count()`、`insert()`、`append()`、`index()`、`remove()`、`reverse()`、`sort()`等。

## Tuple（元组）
元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号`()`里，元素之间用逗号隔开。元组中的元素类型也可以不相同：
```python
tuple = ( 'abcd', 786 , 2.23, 'runoob', 70.2  )
tinytuple = (123, 'runoob')

print(tuple)             # 输出完整元组
print(tuple[0])          # 输出元组的第一个元素
print(tuple[1:3])        # 输出从第二个元素开始到第三个元素
print(tuple[2:])         # 输出从第三个元素开始的所有元素
print(tinytuple * 2)     # 输出两次元组
print(tuple + tinytuple) # 连接元组
```

以上实例输出结果如下：
```python
('abcd', 786, 2.23, 'runoob', 70.2)
abcd
(786, 2.23)
(2.23, 'runoob', 70.2)
(123, 'runoob', 123, 'runoob')
('abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob')
```

元组与字符串类似，可以被索引且下标索引从0开始，-1为从末尾开始的位置。也可以进行截取：
```python
>>> tup = (1, 2, 3, 4, 5, 6)
>>> print(tup[0])
1
>>> print(tup[1:5])
(2, 3, 4, 5)
>>> tup[0] = 11  # 修改元组元素的操作是非法的
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
```

虽然tuple的元素不可改变，但它可以包含可变的对象，比如list列表。构造包含0个或1个元素的元组比较特殊，所以有一些额外的语法规则：
```python
>>> tup1 = ()    # 空元组
>>> tup2 = (20,) # 一个元素，需要在元素后添加逗号
>>> a = [1, 2, 3]  # 可变列表
>>> c = (a, 1)     # 一个元组
>>> print(c)
([1, 2, 3], 1)
>>> a.append("fly")
>>> print(c)
([1, 2, 3, 'fly'], 1)
```

## Set（集合）
集合（set）是一个无序不重复元素的序列。基本功能是进行成员关系测试和删除重复元素。可以使用大括号`{}`或`set()`函数创建集合。注意：创建一个空集合必须用`set()`而不是`{}`，因为`{}`是用来创建一个空字典。
```python
student = {'Tom', 'Jim', 'Mary', 'Tom', 'Jack', 'Rose'}
print(student)   # 输出集合，重复的元素被自动去掉

# 成员测试
if('Rose' in student) :
    print('Rose 在集合中')
else :
    print('Rose 不在集合中')

# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')

print(a)
print(a - b)     # a和b的差集
print(a | b)     # a和b的并集
print(a & b)     # a和b的交集
print(a ^ b)     # a和b中不同时存在的元素
```

以上实例输出结果如下：
```python
{'Jack', 'Rose', 'Mary', 'Jim', 'Tom'}
Rose 在集合中
{'r', 'b', 'a', 'c', 'd'}
{'r', 'b', 'd'}
{'a', 'l', 'z', 'b', 'm', 'd', 'r', 'c'}
{'a', 'c'}
{'l', 'z', 'b', 'm', 'd', 'r'}
```

## Dictionary（字典）
字典（dictionary）是Python中另一个非常有用的内置数据类型。列表是有序的对象结合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。字典是一种映射类型，字典用`{}`标识，它是一个无序的`key:value`对集合。`key`必须使用不可变类型。在同一个字典中，`key`必须是唯一的。
```python
dict = {}
dict['one'] = "1 - 菜鸟教程"
dict[2]     = "2 - 菜鸟工具"

tinydict = {'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}
print(dict['one'])       # 输出键为 'one' 的值
print(dict[2])           # 输出键为 2 的值
print(tinydict)          # 输出完整的字典
print(tinydict.keys())   # 输出所有键
print(tinydict.values()) # 输出所有值
```

以上实例输出结果如下：
```python
1 - 菜鸟教程
2 - 菜鸟工具
{'name': 'runoob', 'site': 'www.runoob.com', 'code': 1}
dict_keys(['name', 'site', 'code'])
dict_values(['runoob', 'www.runoob.com', 1])
```

构造函数`dict()`可以直接从键值对序列中构建字典如下：
```python
>>> dict([('Runoob', 1), ('Google', 2), ('Taobao', 3)])
{'Taobao': 3, 'Runoob': 1, 'Google': 2}
>>> {x: x**2 for x in (2, 4, 6)}
{2: 4, 4: 16, 6: 36}
>>> dict(Runoob=1, Google=2, Taobao=3)
{'Taobao': 3, 'Runoob': 1, 'Google': 2}
```

另外，字典类型也有一些内置的函数，例如`clear()`、`keys()`、`values()`等。

## 数据类型转换
有时候，我们需要对数据内置的类型进行转换，数据类型的转换，你只需要将数据类型作为函数名即可。以下几个内置的函数可以执行数据类型之间的转换。这些函数返回一个新的对象，表示转换的值。
```python
int(x [,base])         # 将x转换为一个整数
float(x)               # 将x转换到一个浮点数
complex(real [,imag])  # 创建一个复数
str(x)                 # 将对象 x 转换为字符串
repr(x)                # 将对象 x 转换为表达式字符串
eval(str)              # 用来计算在字符串中的有效Python表达式,并返回一个对象
tuple(s)               # 将序列 s 转换为一个元组
list(s)                # 将序列 s 转换为一个列表
set(s)                 # 转换为可变集合
dict(d)                # 创建一个字典。d 必须是一个序列 (key,value) 元组。
frozenset(s)           # 转换为不可变集合
chr(x)                 # 将一个整数转换为一个字符
unichr(x)              # 将一个整数转换为Unicode字符
ord(x)                 # 将一个字符转换为它的整数值
hex(x)                 # 将一个整数转换为一个十六进制字符串
oct(x)                 # 将一个整数转换为一个八进制字符串
```

## 参考资料：
- [Python 3 教程](http://www.runoob.com/python3/python3-tutorial.html)