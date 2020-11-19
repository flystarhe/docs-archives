# Call matlab
[Call MATLAB Functions from Python](https://ww2.mathworks.cn/help/matlab/matlab_external/call-matlab-functions-from-python.html?lang=en).

## Install MATLAB Engine API for Python
On Windows systems:
```
cd matlab_root\extern\engines\python
python setup.py install
```

On Mac or Linux systems:
```
cd matlab_root/extern/engines/python
python setup.py install
```

## Return Output Argument from MATLAB Function
```python
import matlab.engine
eng = matlab.engine.start_matlab()
tf = eng.isprime(37)
print(tf)
#True
```

## Return Multiple Output Arguments from MATLAB Function
```python
import matlab.engine
eng = matlab.engine.start_matlab()
t = eng.gcd(100.0, 80.0, nargout=3)
print(t)
#(20.0, 1.0, -1.0)
```

## Call MATLAB Functions Asynchronously from Python
```python
import matlab.engine
eng = matlab.engine.start_matlab()
future = eng.sqrt(4.0, async=True)
ret = future.result()
print(ret)
#2.0
```

## Call User Script and Function from Python
在您的当前文件夹中名为`triarea.m`的文件中:
```python
function a = triarea(b,h)
a = 0.5*(b.* h);
```

启动Python调用该脚本:
```python
import matlab.engine
eng = matlab.engine.start_matlab()
ret = eng.triarea(1.0, 5.0)
print(ret)
#2.5
```

## Create MATLAB Arrays in Python
```python
import matlab.engine
eng = matlab.engine.start_matlab()

A = matlab.int8([1,2,3,4,5])
print(A.size)
#(1, 5)

A = matlab.double([[1,2,3,4,5], [6,7,8,9,10]])
print(A)
#[[1.0,2.0,3.0,4.0,5.0],[6.0,7.0,8.0,9.0,10.0]]

import matlab.engine
A = matlab.int8([1,2,3,4,5,6,7,8,9])
A.reshape((3,3))
print(A)
#[[1,4,7],[2,5,8],[3,6,9]]
```

## userpath
```python
import matlab.engine
eng = matlab.engine.start_matlab()

eng.cd()   #显示当前文件夹
eng.pwd()  #显示当前文件夹
eng.path() #显示搜索路径

eng.userpath() #返回搜索路径上的第一个文件夹
eng.userpath("newpath", nargout=0) #修改搜索路径上的第一个文件夹
eng.userpath("reset", nargout=0)   #设置搜索路径上的第一个文件夹为您的平台的默认文件夹
eng.userpath("clear", nargout=0)   #删除搜索路径中的第一个文件夹

eng.exit()
```

>`oldFolder = eng.cd(newFolder)`将现有的当前文件夹返回给`oldFolder`,然后将当前文件夹更改为`newFolder`.