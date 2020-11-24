# Concurrent futures
`concurrent.futures`模块提供异步执行回调高层接口。异步执行可以由`ThreadPoolExecutor`使用线程或由`ProcessPoolExecutor`使用单独的进程来实现。两者都是实现抽像类`Executor`定义的接口。

## Executor
执行器对象。抽象类提供异步执行调用方法。要通过它的子类调用，而不是直接调用。

- `submit(fn, *args, **kwargs)`：调度可调用对象`fn`，以`fn(*args, **kwargs)`方式执行并返回`Future`对像代表可调用对象的执行。
- `map(func, *iterables, timeout=None, chunksize=1)`：`func`是异步执行的且对`func`的调用可以并发执行。使用`ProcessPoolExecutor`时，这个方法会将`iterables`分割任务块并作为独立的任务并提交到执行池中。这些块的大概数量可以由`chunksize`指定正整数设置。对很长的迭代器来说，使用大的`chunksize`值比默认值1能显著地提高性能。`chunksize`对`ThreadPoolExecutor`没有效果。
- `shutdown(wait=True)`：当待执行的期程完成执行后向执行者发送信号，它就会释放正在使用的任何资源。如果`wait`为`True`则此方法只有在所有待执行的期程完成执行且释放已分配的资源后才会返回。如果`wait`为`False`，方法立即返回，所有待执行的期程完成执行后会释放已分配的资源。不管`wait`的值是什么，整个Python程序将等到所有待执行的期程完成执行后才退出。如果使用`with`语句，你就可以避免显式调用这个方法。

## ThreadPoolExecutor
`ThreadPoolExecutor`是`Executor`的子类，它使用线程池来异步执行调用。使用最多`max_workers`个线程的线程池来异步执行调用。

```python
import concurrent.futures
import urllib.request

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))
```

## ProcessPoolExecutor
`ProcessPoolExecutor`是`Executor`的子类，它使用进程池来实现异步执行调用。`ProcessPoolExecutor`使用`multiprocessing`回避`Global Interpreter Lock`但也意味着只可以处理和返回可序列化的对象。`__main__`模块必须可以被工作者子进程导入。这意味着`ProcessPoolExecutor`不可以工作在交互式解释器中。

```python
import concurrent.futures
import math

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

if __name__ == '__main__':
    main()
```

## Future
`Future`类将可调用对象封装为异步执行。`Future`实例由`Executor.submit()`创建。

- `cancel()`：尝试取消调用。如果调用正在执行而且不能被取消那么方法返回False，否则调用会被取消同时方法返回True。
- `cancelled()`：如果调用成功取消返回True。
- `running()`：如果调用正在执行而且不能被取消那么返回True。
- `done()`：如果调用已被取消或正常结束那么返回True。
- `result(timeout=None)`：返回调用返回的值。如果调用还没完成那么这个方法将等待`timeout`秒。如果在`timeout`秒内没有执行完成，`concurrent.futures.TimeoutError`将会被触发。
- `exception(timeout=None)`：返回由调用引发的异常。如果调用还没完成那么这个方法将等待`timeout`秒。如果在`timeout`秒内没有执行完成，`concurrent.futures.TimeoutError`将会被触发。
- `add_done_callback(fn)`：附加可调用`fn`到期程。当期程被取消或完成运行时，将会调用`fn`，而这个期程将作为它唯一的参数。

## 参考资料：
- [启动并行任务](https://docs.python.org/zh-cn/3/library/concurrent.futures.html)