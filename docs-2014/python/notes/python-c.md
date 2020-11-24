# C to Python
绝大多数时候,C表示高执行性能,Python表示高开发效率.

## cffi(API level, out-of-line)
cffi提供了`ABI level`和`API level`两种模式,API模式比ABI模式更快.
```python
# file "example_build.py"
from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("""
int sum(int i, int j);
""")

ffibuilder.set_source("_example", """
int sum(int i, int j)
{
    return 1000+i+j;
}
""")

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
```

您需要运行`example_build.py`脚本,生成源代码到文件`_example.c`,并编译为常规的C扩展模块`_example.so`或`_example.pyd`.

然后,在Python主程序中使用:
```python
# file "example.py"
from _example import ffi, lib

lib.sum(1, 11)
## 1012
```

它是`API级别`,而不是`ABI级别`.它需要一个C编译器才能运行,但它更方便.还要注意,在运行时,API模式比ABI模式更快.

## swig
```bash
sudo apt-get install swig   # Linux
brew install swig           # OS X with homebrew
```

假设你有一些C函数,你想添加到Tcl,Perl,Python,Java. [
Tutorial](http://www.swig.org/tutorial.html)
```c
/* File : example.c */

#include <time.h>
double My_variable = 3.0;

int fact(int n) {
    if (n <= 1) return 1;
    else return n*fact(n-1);
}

int my_mod(int x, int y) {
    return (x%y);
}

char *get_time()
{
    time_t ltime;
    time(&ltime);
    return ctime(&ltime);
}
```

为了将这些添加到您喜欢的语言中,您需要编写一个`接口文件`,它是SWIG的输入:
```
/* example.i */
%module example
%{
/* Put header files here or function declarations like below */
extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
%}

extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
```

将C代码转换为Python模块也很容易,只需执行以下操作:
```
unix % swig -python example.i
unix % gcc -c example.c example_wrap.c -I/usr/local/include/python2.1
unix % ld -shared example.o example_wrap.o -o _example.so 
```

现在可以使用Python模块:
```python
>>> import example
>>> example.fact(5)
120
>>> example.my_mod(7,3)
1
>>> example.get_time()
'Sun Feb 11 23:01:07 1996'
```

## Installing Bazel
[ubuntu](https://docs.bazel.build/versions/master/install-ubuntu.html):
```bash
sudo apt-get install openjdk-8-jdk
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel
```

[mac](https://docs.bazel.build/versions/master/install-os-x.html) and [Oracle's JDK Page](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html):
```bash
/usr/bin/ruby -e "$(curl -fsSL \
https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install bazel
bazel version
brew upgrade bazel
```
