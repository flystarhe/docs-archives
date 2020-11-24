title: Python内置函数清单
date: 2017-10-10
tags: [Python]
---
Python内置函数随着python解释器的运行而创建,你可以随时调用这些函数,不需要定义.最常见的内置函数是`print("Hello World!")`.

<!--more-->
## 常用函数
`type()`,`dir()`,`help()`,`len()`,`open()`,`range()`,`enumerate()`,`zip()`,`iter()`,`map()`,`filter()`,`reduce()`,`id`.

## 数学运算
```python
abs(-5)                     # 取绝对值,也就是5
round(2.6)                  # 四舍五入取整,也就是3.0
pow(2, 3)                   # 相当于2**3,如果是pow(2, 3, 5),相当于(2**3%5)
cmp(2.3, 3.2)               # 比较两个数的大小
divmod(9, 2)                # 返回除法结果和余数
max([1, 5, 2, 9])           # 求最大值
min([9, 2, 4, 2])           # 求最小值
sum([2, 1, 9, 12])          # 求和
```

## 类型转换
```python
int("5")                         # 转换为整数 integer
float(2)                         # 转换为浮点数 float
long("23")                       # 转换为长整数 long integer
str(2.3)                         # 转换为字符串 string
complex(3, 9)                    # 返回复数 3 + 9i
ord("A")                         # "A"字符对应的数值
chr(65)                          # 数值65对应的字符
bool(0)                          # 转换为相应的真假值
#下列对象都相当于False: [],(),{},0,None,0.0,''
bin(56)                          # 返回一个字符串,表示56的二进制数
hex(56)                          # 返回一个字符串,表示56的十六进制数
oct(56)                          # 返回一个字符串,表示56的八进制数
list((1, 2, 3))                  # 转换为表 list
tuple([2, 3, 4])                 # 转换为定值表 tuple
slice(5, 2, 1)                   # 构建下标对象 slice
dict(a=1, b="hello", c=[1,2,3])  # 构建词典 dictionary
```

## 序列操作
```python
all([True, 1, "hello!"])         # 是否所有的元素都相当于True值
any(["", 0, False, [], None])    # 是否有任意一个元素相当于True值
sorted([1, 5, 3])                # 返回正序的序列,也就是([1,3,5])
reversed([1, 5, 3])              # 返回反序的序列,也就是([3,5,1])
```

## 类,对象,属性
```python
# define class
class Me(object):
    def test(self):
        print("Hello!")

def new_test():
    print("New Hello!")

me = Me()
hasattr(me, "test")               # 检查me对象是否有test属性
getattr(me, "test")               # 返回test属性
setattr(me, "test", new_test)     # 将test属性设置为new_test
delattr(me, "test")               # 删除test属性
isinstance(me, Me)                # me对象是否为Me类生成的对象
issubclass(Me, object)            # Me类是否为object类的子类
```

## 编译,执行
```python
repr(me)                 # 返回对象的字符串表达
compile("print('Hello')", 'test.py', 'exec')  # 编译字符串成为code对象
eval("1 + 1")            # 解释字符串表达式,参数也可以是compile()返回的code对象
exec("print('Hello')")   # 解释并执行字符串,参数也可以是compile()返回的code对象
```
