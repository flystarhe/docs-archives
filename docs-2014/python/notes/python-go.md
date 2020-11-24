# Go to Python
Go可以编译生成`.so`对象,生成的`.so`文件可以被Python使用.继续阅读,会看到一些有趣的代码.

## Build
`main.go`:
```go
package main

import "C"

//export Add
func Add(i, j int) int {
    return i + j
}

func main() {}
```

在函数定义之前必须写入`//export FUNCNAME`.

执行`go build -buildmode=c-shared -o add.so main.go`,会得到`add.so`和`add.h`.

## Usage

### ctypes
在Python代码中调用`Add()`:
```python
from ctypes import cdll

libc = cdll.LoadLibrary("./add.so")
libc.Add(1, 2)
```

### cffi
在Python代码中调用`Add()`:
```python
from cffi import FFI

ffi = FFI()

ffi.cdef("""
int Add(int i, int j);
""")

libc = ffi.dlopen("./add.so")
print(libc.Add(11, 12))
```

## Other

- [github: go-python/gopy](https://github.com/go-python/gopy)
- [github: sbinet/go-python](https://github.com/sbinet/go-python)
- [python调用Go代码](http://blog.csdn.net/yhcharles/article/details/48154143)

## Reference
- [BUILDING PYTHON MODULES WITH GO 1.5](https://blog.filippo.io/building-python-modules-with-go-1-5/)