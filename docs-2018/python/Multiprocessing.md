# Multiprocessing
`multiprocessing`是一个用与`threading`模块相似API的支持产生进程的包。`multiprocessing`包同时提供本地和远程并发，使用子进程代替线程，有效避免Global Interpreter Lock带来的影响。因此，`multiprocessing`模块允许程序员充分利用机器上的多个核心。

## Pool
它提供了一种方便的方法，可以跨多个输入值并行化函数的执行，跨进程分配输入数据（数据并行）。
```python
import time
from multiprocessing import Pool

def f(x):
    import os
    return "++{}++{}++{}++".format(x, os.getpid(), os.getppid())

if __name__ == "__main__":
    with Pool(2) as p:
        print(p.map(f, [1, 2, 3, 4, 5]))
```

一个进程池对象，它控制可以提交作业的工作进程池。它支持带有超时和回调的异步结果，并具有并行映射实现。

- `apply(func[, args[, kwds]])`：它会阻塞，直到结果准备就绪。`apply_async`更适合并行执行工作。
- `apply_async(func[, args[, kwds[, callback[, error_callback]]]])`：它返回一个结果对象。如果指定了回调，则它应该是可调用的，它接受单个参数。当结果变为就绪时，将对其应用回调，即除非调用失败，在这种情况下将应用`error_callback`。
- `map(func, iterable[, chunksize])`：它会阻塞，直到结果准备就绪。此方法将可迭代切换为多个块，并将其作为单独的任务提交给进程池。
- `map_async(func, iterable[, chunksize[, callback[, error_callback]]])`：它返回一个结果对象。如果指定了回调，则它应该是可调用的，它接受单个参数。当结果变为就绪时，将对其应用回调，即除非调用失败，在这种情况下将应用`error_callback`。
- `close()`：防止将任何其他任务提交到池中。完成所有任务后，工作进程将退出。
- `join()`：等待工作进程退出。必须在使用之前调用`close`或`terminate`。

## AsyncResult
`Pool.apply_async`和`Pool.map_async`返回的结果类。

- `get([timeout])`：到达时返回结果。如果`timeout`不是`None`并且结果未在超时秒内到达，则引发`multiprocessing.TimeoutError`。
- `wait([timeout])`：等到结果可用或直到超时秒数过去。
- `ready()`：返回调用是否已完成。
- `successful()`：返回是否完成调用而不引发异常。

## 演示池的使用
```python
from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    return x*x

if __name__ == "__main__":
    # start 4 worker processes
    with Pool(processes=4) as pool:
        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        it = pool.imap(f, range(10))
        print(next(it))                     # prints "0"
        print(next(it))                     # prints "1"
        print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))      # runs in *only* one process
        print(res.get(timeout=1))             # prints "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        print(res.get(timeout=1))             # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
```

## 参考资料：
- [基于进程的并行](https://docs.python.org/zh-cn/3/library/multiprocessing.html)