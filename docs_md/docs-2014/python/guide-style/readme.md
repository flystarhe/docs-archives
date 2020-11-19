title: Python Language and Style Rules
date: 2017-07-10
tags: [Python]
---
Python语言与风格规范.包含:行长度,文档编排,文档描述,空格使用,命名约定,编码建议.

<!--more-->
## 行长度
每行不超过80个字符,仅长的导入模块语句和注释里的URL允许例外.也不要使用`\`连接行.如果一个文本字符串在一行放不下,可以使用圆括号来实现隐式行连接:
```
x = ('这是一个很长的字符串'
     '真的真的很长')
```

## 文档编排
顶级定义之间空两行,比如函数或者类定义.方法定义之间空一行.函数或者方法中,某些地方要是你觉得合适,就空一行,以增加代码块的可读性.

模块内容的顺序:
```
模块说明

docstring

import 标准

import 三方

import 自己

globals + constants + 其他定义
```

不要`import`多个库:
```
Yes: import os
     import sys
No:  import os, sys
```

用4个空格来缩进代码,绝不要用tab,更不要tab和空格混用.对于行连接的情况,要么垂直对齐换行元素,要么使用4个空格的悬挂式缩进.

垂直对齐换行元素:(推荐)
```
foo = long_function_name(var_one, var_two,
                         var_three, var_four)
```

4个空格的悬挂式缩进:
```
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)
```

## 文档描述
为所有的共有模块,函数,类,方法写`docstring`,非共有的没有必要,但是可以写注释.

块注释,在一段代码前增加的注释,在`#`后加一空格,段落之间以只有`#`的行间隔:
```
# Description : Module config.
# 
# Input : None
#
# Output : None
```

行注释,在一句代码后加注释,在`#`后加一空格:(尽量避免使用)
```
x = x + 1 # Increment x
```

## 空格使用
按照标准的排版规范来使用标点两边的空格:
```
Yes: spam(ham[1], {eggs: 2}, [])
```
```
No:  spam( ham[ 1 ], { eggs: 2 }, [ ] )
```

不要在逗号,分号,冒号前面使用空格,但应该在它们后面加(行尾除外):
```
Yes: if x == 4:
         print x, y
     x, y = y, x
```
```
No:  if x == 4 :
         print x , y
     x , y = y , x
```

参数列表,索引或切片的左括号前不要加空格:
```
Yes: spam(1)
```
```
No:  spam (1)
```

在二元操作符两边都加上一个空格,比如赋值(=),比较(==,!=,in,not in,is,is not),布尔(and,or,not).至于算术操作符两边的空格该如何使用,需要你自己好好判断,不过两侧务必要保持一致:
```
Yes: x == 1
```
```
No:  x<1
```

当`=`用于指示关键字参数或默认参数值时,不要在其两侧使用空格:
```
Yes: def complex(real, imag=0.0): return magic(r=real, i=imag)
```

## 命名约定
应该避免:

- 单字符名称,除了计数器和迭代器
- 包/模块名中的连字符`-`
- 双下划线开头并结尾的名称(Python保留)

约定:

- 包命名尽量短小,使用全部小写的方式,不可以使用下划线
- 模块命名尽量短小,使用全部小写的方式,可以使用下划线
- 类的命名使用`CapWords`的方式,模块内部使用的类采用`_CapWords`的方式
- 函数命名使用全部小写的方式,可以使用下划线
- 常量命名使用全部大写的方式,可以使用下划线
- 类的属性(方法和变量)命名使用全部小写的方式,可以使用下划线
- 类的属性若与关键字名字冲突,后缀一下划线,尽量不要使用缩略等其他方式
- 类的方法第一个参数必须是self,而静态方法第一个参数必须是cls

## 编码建议
使用`isinstance()`比较对象的类型:
```
Yes: if isinstance(obj, int):
No:  if type(obj) is type(1):
```

判断序列空或不空,有如下规则:
```
Yes: if not seq:
Yes: if seq:
No:  if len(seq)
No:  if not len(seq)
```

尽可能使用`is/is not`取代`==`:
```
Yes: if x is not None
No:  if x
```

如果一个类不继承自其它类,就显式的从object继承.嵌套类也一样:
```
class SampleClass(object):
    pass


class OuterClass(object):

    class InnerClass(object):
        pass


class ChildClass(ParentClass):
    """Explicitly inherits from another class already."""
    pass
```

## 参考资料:
- [Python语言规范](http://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_language_rules/)
- [Python风格规范](http://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)
- [Python编码规范](http://www.runoob.com/w3cnote/google-python-styleguide.html)