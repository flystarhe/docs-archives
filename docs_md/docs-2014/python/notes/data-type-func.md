title: Python数据类型（函数）
date: 2016-10-13
tags: [Python,数据类型]
---
Python3中有六个标准的数据类型：Number（数字），String（字符串），List（列表），Tuple（元组），Sets（集合），Dictionary（字典）。

<!--more-->
## Number（数字）
数学函数：
```python
abs(x)     # 返回数字的绝对值，如abs(-10) 返回 10
ceil(x)    # 返回数字的上入整数，如math.ceil(4.1) 返回 5
exp(x)     # 返回e的x次幂，如math.exp(1) 返回 2.718281828459045
fabs(x)    # 返回数字的绝对值，如math.fabs(-10) 返回 10.0
floor(x)   # 返回数字的下舍整数，如math.floor(4.9) 返回 4
log(x)     # 如math.log(math.e) 返回 1.0，math.log(100, 10) 返回 2.0
log10(x)   # 返回以10为基数的x的对数，如math.log10(100) 返回 2.0
max(x1, x2, ..) # 返回给定参数的最大值，参数可以为序列。
min(x1, x2, ..) # 返回给定参数的最小值，参数可以为序列。
modf(x)         # 返回x的整数部分与小数部分。
pow(x, y)       # x**y 运算后的值。
round(x [,n])   # 返回浮点数x的四舍五入值，n代表舍入到小数点后的位数。
sqrt(x)         # 返回数字x的平方根。
```

随机函数：
```python
choice(seq)     # 从序列中随机挑选一个元素，如random.choice(range(10))
randrange ([start,] stop [,step])  # 从指定范围内获取一个随机数
random()        # 随机生成下一个实数，它在[0,1)范围内
sample()        # 不放回抽样
seed([x])       # 改变随机数生成器的种子seed
shuffle(lst)    # 将序列的所有元素随机排序
uniform(x, y)   # 随机生成下一个实数，它在[x,y]范围内
```

三角函数：
```python
acos(x)     # 返回x的反余弦弧度值。
asin(x)     # 返回x的反正弦弧度值。 
atan(x)     # 返回x的反正切弧度值。
atan2(y, x) # 返回给定的 X 及 Y 坐标值的反正切值。
cos(x)      # 返回x的弧度的余弦值。
hypot(x, y) # 返回欧几里德范数 sqrt(x*x + y*y)。
sin(x)      # 返回的x弧度的正弦值。
tan(x)      # 返回x弧度的正切值。
degrees(x)  # 将弧度转换为角度，如degrees(math.pi/2) 返回 90.0
radians(x)  # 将角度转换为弧度
```

## String（字符串）
字符串运算符：
```python
+       # 字符串连接 a+b
*       # 重复输出字符串 a*2
[]      # 通过索引获取字符串中字符 a[1]
[ : ]   # 截取字符串中的一部分 a[1:4]
in      # 成员运算符 "H" in "Hello"
not in  # 成员运算符
r/R     # 原始字符串 print r'\\n'
%       # 格式字符串
```

字符串格式化：
```python
>>> print("我叫 %s 今年 %d 岁!" % ('小明', 10))  # 推荐新式 str.format
我叫 小明 今年 10 岁!
%c     # 格式化字符及其ASCII码
%s     # 格式化字符串
%d     # 格式化整数
%u     # 格式化无符号整型
%o     # 格式化无符号八进制数
%x     # 格式化无符号十六进制数
%X     # 格式化无符号十六进制数（大写）
%f     # 格式化浮点数字，可指定小数点后的精度
%e     # 用科学计数法格式化浮点数
%E     # 作用同%e，用科学计数法格式化浮点数
%g     # %f 和 %e 的简写
%G     # %f 和 %E 的简写
%p     # 用十六进制数格式化变量的地址
```

字符串内建函数：
```python
capitalize()   # 将字符串的第一个字符转换为大写
center(width)  # 返回一个指定宽度 width 居中的字符串
count(str)     # 返回 str 在 string 里面出现的次数
decode(encoding='UTF-8')  # 使用指定编码来解码字符串
encode(encoding='UTF-8')  # 以指定的编码格式编码字符串
endswith(suffix)          # 检查字符串是否以 suffix 结束
expandtabs(tabsize=8)     # 把字符串 string 中的 tab 符号转为空格
find(str)    # 检测 str 是否包含在字符串中，如果是返回开始的索引值，否则返回-1
index(str)   # 跟 find() 一样，不过 str 不在字符串中会报一个异常
isalnum()    # 如果字符串至少有一个字符并且所有字符都是字母或数字
isalpha()    # 如果字符串至少有一个字符并且所有字符都是字母
isdigit()    # 如果字符串只包含数字
islower()    # 如果字符串中包含至少一个区分大小写的字符，并且这些字符小写
isnumeric()  # 如果字符串中只包含数字字符
isspace()    # 如果字符串中只包含空格
istitle()    # 如果字符串是标题化的
isupper()    # 如果字符串中包含至少一个区分大小写的字符，并且所有这些大写
join(seq)    # 以指定字符串作为分隔符，将 seq 中所有的元素连接
len(string)  # 返回字符串长度
ljust(width) # 返回一个指定宽度 width 居左的字符串
lower()      # 转换字符串中所有大写字符为小写
lstrip()     # 截掉字符串左边的空格
maketrans()  # 创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。
max(str)     # 返回字符串 str 中最大的字母
min(str)     # 返回字符串 str 中最小的字母
replace(old, new)  # 将字符串中的 str1 替换成 str2
rfind(str)   # 类似于 find()，不过是从右边开始查找
rindex(str)  # 类似于 index()，不过是从右边开始查找
rjust(width) # 返回一个指定宽度 width 居右的字符串
rstrip()     # 删除字符串右边的空格
split(str="")   # 以 str 为分隔符截取字符串
startswith(str) # 检查字符串是否是以 str 开头
strip([chars])  # 在字符串上执行 lstrip() 和 rstrip()
title()         # 返回"标题化"的字符串，所有单词都是以大写开始，其余字母小写
upper()         # 转换字符串中的小写字母为大写
zfill(width)    # 返回长度为 width 的字符串，原字符串右对齐，前面填充0
```

## List（列表）
访问列表中的值，更新列表，删除列表元素：
```python
>>> list1 = ['Google', 'Runoob', 1997, 2000]
>>> print(list1[0])
Google
>>> list1[2] = 2001
>>> print(list1)
['Google', 'Runoob', 2001, 2000]
>>> del list1[2]
>>> print(list1)
['Google', 'Runoob', 2000]
```

列表函数：
```python
len(list)  # 列表元素个数
max(list)  # 返回列表元素最大值
min(list)  # 返回列表元素最小值
list(seq)  # 将元组转换为列表
```

列表方法：
```python
list.append(obj)    # 在列表末尾添加新的对象
list.count(obj)     # 统计某个元素在列表中出现的次数
list.extend(seq)    # 在列表末尾一次性追加另一个序列中的多个值
list.index(obj)     # 从列表中找出某个值第一个匹配项的索引位置
list.insert(index, obj)  # 将对象插入列表
list.pop(obj=list[-1])   # 移除列表中的一个元素（默认最后一个），并且返回该元素的值
list.remove(obj)    # 移除列表中某个值的第一个匹配项
list.reverse()      # 反向列表中元素
list.sort([func])   # 对原列表进行排序
list.clear()        # 清空列表
list.copy()         # 复制列表
```

## Tuple（元组）
元组函数：
```python
len(tuple)  # 计算元组元素个数
max(tuple)  # 返回元组中元素最大值
min(tuple)  # 返回元组中元素最小值
tuple(seq)  # 将列表转换为元组
```

## Sets（集合）
pass

## Dictionary（字典）
访问字典里的值，修改字典，删除字典元素：
```python
dict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
print("dict['Name']: ", dict['Name'])  # 访问字典里的值
dict['Age'] = 8;             # 更新 Age
dict['School'] = "菜鸟教程"  # 添加信息
del dict['Name'] # 删除键 'Name'
dict.clear()     # 删除字典
del dict         # 删除字典
```

字典函数：
```python
len(dict)   # 计算字典元素个数，即键的总数
str(dict)   # 输出字典以可打印的字符串表示
```

字典方法：
```python
radiansdict.clear()  # 删除字典内所有元素
radiansdict.copy()   # 返回一个字典的浅复制
radiansdict.get(key) # 返回指定键的值，如果值不在字典中返回default值
key in dict          # 如果键在字典dict里返回true，否则返回false
radiansdict.items()  # 以列表返回可遍历的 (key, val) 元组数组
radiansdict.keys()   # 以列表返回一个字典所有的键
radiansdict.setdefault(key, default=None)  # 和get()类似，但如果键不存在于字典中，将会添加键并将值设为default
radiansdict.update(dict2)  # 把字典dict2的键/值对更新到dict里
radiansdict.values()       # 以列表返回字典中的所有值
```

## 参考资料：
- [Python 3 教程](http://www.runoob.com/python3/python3-tutorial.html)