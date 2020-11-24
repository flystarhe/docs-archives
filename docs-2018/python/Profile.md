# Profile
为了更好了解`Python`程序,我们需要一套工具,能够记录代码运行时间,生成一个性能分析报告,方便彻底了解代码,从而进行针对性的优化.标准库里面的`profile`或者`cProfile`,它可以统计程序里每一个函数的运行时间,并且提供了可视化的报表.

## 关于性能分析
性能分析就是分析代码和正在使用的资源之间有着怎样的联系,它可以帮助我们分析运行时间从而找到程序运行的瓶颈,也可以帮助我们分析内存的使用防止内存泄漏的发生.帮助我们进行性能分析的工具便是性能分析器,它主要分为两类:

- 基于事件的性能分析(event-based profiling)
- 统计式的性能分析(statistical profiling)

## Python的性能分析器
Python中最常用的性能分析工具主要有:`cProfile`,`line_profiler`以及`memory_profiler`等.他们以不同的方式帮助我们分析Python代码的性能.我们这里主要关注Python内置的`cProfile`,并使用它帮助我们分析并优化程序.

## cProfile
官方文档的一个简单例子:
```python
import cProfile
import re
cProfile.run('re.compile("foo|bar")')
```

分析结果:
```
         185 function calls (180 primitive calls) in 0.000 seconds

   Ordered by: standard name

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.000    0.000 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 re.py:222(compile)
        1    0.000    0.000    0.000    0.000 re.py:278(_compile)
        1    0.000    0.000    0.000    0.000 sre_compile.py:221(_compile_charset)
        1    0.000    0.000    0.000    0.000 sre_compile.py:248(_optimize_charset)
        1    0.000    0.000    0.000    0.000 sre_compile.py:412(_compile_info)
        2    0.000    0.000    0.000    0.000 sre_compile.py:513(isstring)
        1    0.000    0.000    0.000    0.000 sre_compile.py:516(_code)
        ..
```

从分析报告结果中我们可以得到很多信息:

1. 总共执行的时间为0.000秒
2. 整个过程一共有185个函数调用被监控,其中180个是原生调用,即不涉及递归调用
3. 结果列表中是按照标准名称进行排序,也就是按照字符串的打印方式,数字也当作字符串

- ncalls:表示函数调用的次数,有两个数值表示有递归调用`总调用次数/原生调用次数`
- tottime:是函数内部调用时间,不包括他自己调用的其他函数的时间
- cumtime:累积调用时间,它包含了自己内部调用函数的时间
- percall:等于`tottime/ncalls`

### 优雅的使用
Python给我们提供了很多接口方便我们能够灵活的进行性能分析,其中主要包含`cProfile`模块的`Profile`类和`pstat`模块的`Stats`类.

Profile类:

- `enable()`:开始收集性能分析数据
- `disable()`:停止收集性能分析数据
- `create_stats()`:停止收集分析数据,并为已收集的数据创建`Stats`对象
- `print_stats()`:创建`Stats`对象并打印分析结果
- `dump_stats(filename)`:把当前性能分析的结果写入文件
- `run(cmd)`:收集被执行命令`cmd`的性能分析数据
- `runcall(func, *args, **kwargs)`:收集被调用函数`func`的性能分析数据

Stats类:

- `strip_dirs()`:删除报告中所有函数文件名的路径信息
- `dump_stats(filename)`:把stats中的分析数据写入文件,效果同`cProfile.Profile.dump_stats()`
- `sort_stats(*keys)`:对报告列表进行排序,具体参数参见[doc](https://docs.python.org/3.6/library/profile.html#pstats.Stats)
- `reverse_order()`:逆反当前的排序
- `print_stats(*restrictions)`:把信息打印到标准输出,`*restrictions`用于控制打印结果的形式

有了上面的接口我们便可以更优雅的去使用分析器来分析我们的程序,例如可以通过写一个带有参数的装饰器,这样想分析项目中任何一个函数,便可方便的使用装饰器来达到目的:
```python
import cProfile
import pstats
import os
# 性能分析装饰器定义
def do_cprofile(filename):
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            DO_PROF = os.getenv("PROFILING")
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper
```

这样我们可以在我们想进行分析的地方进行性能分析,例如我想分析`MyClass`类中的`run`方法:

```python
# run.py
class MyClass(object):
    # 应用装饰器来分析函数
    @do_cprofile("./myclass_run.prof")
    def run(self, **kwargs):
        # ...
```

装饰器函数中通过`sys.getenv`来获取环境变量判断是否需要进行分析,因此可以通过设置环境变量来告诉程序是否进行性能分析:

```bash
export PROFILING=y
python run.py
```

程序跑完后便会在当前路径下生成`myclass_run.prof`的分析文件,我们便可以通过打印或者可视化工具来对这个函数进行分析.用pstats模块的接口来读取:

```python
p = pstats.Stats('myclass_run.prof')
p.strip_dirs().sort_stats('cumtime').print_stats(10, 1.0, '.*')
```

## gprof2dot
[jrfonseca/gprof2dot](https://github.com/jrfonseca/gprof2dot),`Debian/Ubuntu/Mac`依赖`apt-get/brew install graphviz`,`pip install graphviz`,执行`pip install gprof2dot`完成安装.`example.py`内容如下:
```python
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()
```

生成分析图:

    $ python -m cProfile -o output.pstats example.py
    $ git clone https://github.com/jrfonseca/gprof2dot.git
    $ cd gprof2dot/
    $ ./gprof2dot.py -f pstats ../output.pstats | dot -Tpng -o output.png

## vprof
[nvdv/vprof](https://github.com/nvdv/vprof),执行`pip install vprof`完成安装.针对文件进行执行并分析,并在浏览器中生成可视化图标:

    $ vprof -c chmp example.py

## line_profiler
`cProfile`只能返回函数整体的耗时.[rkern/line_profiler](https://github.com/rkern/line_profiler),执行`pip install line_profiler`完成安装.在下面的例子中,我们创建一个简单的函数`my_func`来分配列表`a,b`然后删除`b`:
```python
@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()
```

命令行执行:

    $ kernprof -l example.py
    $ python -m line_profiler example.py.lprof

输出结果如下:
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     2                                           @profile
     3                                           def my_func():
     4         1         6311   6311.0      3.7      a = [1] * (10 ** 6)
     5         1       104459 104459.0     60.7      b = [2] * (2 * 10 ** 7)
     6         1        61362  61362.0     35.6      del b
     7         1            9      9.0      0.0      return a
```

### 优雅的使用
加`@profile`后函数无法直接运行,只能优化的时候加上,调试的时候又得去掉.实用的方法是:
```python
from line_profiler import LineProfiler

def do_other_stuff(numbers):
    s = sum(numbers)
    return s

def do_stuff(numbers):
    s = do_other_stuff(numbers)
    return s

if __name__ == '__main__':
    numbers = [i for i in range(9)]
    lp = LineProfiler()
    lp_wrapper = lp(do_stuff)
    lp_wrapper(numbers)
    lp.print_stats()
```

执行`python example.py`,输出结果如下:
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     7                                           def do_stuff(numbers):
     8         1            7      7.0     87.5      s = do_other_stuff(numbers)
     9         1            1      1.0     12.5      return s
```

### 函数内部调用函数
这样做的话,只能显示子函数的总时间.为了能够显示调用函数每行所用时间,加入`add_function`:
```python
from line_profiler import LineProfiler

def do_other_stuff(numbers):
    s = sum(numbers)
    return s

def do_stuff(numbers):
    s = do_other_stuff(numbers)
    return s

if __name__ == '__main__':
    numbers = [i for i in range(9)]
    lp = LineProfiler()
    lp.add_function(do_other_stuff)
    lp_wrapper = lp(do_stuff)
    lp_wrapper(numbers)
    lp.print_stats()
```

执行`python example.py`,输出结果如下:
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     3                                           def do_other_stuff(numbers):
     4         1            1      1.0     50.0      s = sum(numbers)
     5         1            1      1.0     50.0      return s
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     7                                           def do_stuff(numbers):
     8         1           11     11.0     91.7      s = do_other_stuff(numbers)
     9         1            1      1.0      8.3      return s
```

## memory_profiler
[fabianp/memory_profiler](https://github.com/fabianp/memory_profiler),执行`pip install memory_profiler`完成安装.在下面的例子中,我们创建一个简单的函数`my_func`来分配列表`a,b`然后删除`b`:
```python
from memory_profiler import profile

@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()
```

执行`python example.py`,输出结果如下:
```
Line #    Mem usage    Increment   Line Contents
================================================
     1   40.086 MiB    0.000 MiB   @profile
     2                             def my_func():
     3   47.727 MiB    7.641 MiB       a = [1] * (10 ** 6)
     4  200.316 MiB  152.590 MiB       b = [2] * (2 * 10 ** 7)
     5   47.727 MiB -152.590 MiB       del b
     6   47.727 MiB    0.000 MiB       return a
```

## 参考资料:
- [Lib/timeit.py](https://docs.python.org/3.6/library/timeit.html)
- [Lib/profile.py](https://docs.python.org/3.6/library/profile.html)
- [line_profiler](https://github.com/rkern/line_profiler)
- [memory_profiler](https://github.com/fabianp/memory_profiler)