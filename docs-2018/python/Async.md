# Async
[异步和非阻塞I/O](http://www.tornadoweb.org/en/stable/guide/async.html)。实时Web功能需要每个用户长期保持大部分空闲连接。在传统的同步Web服务器中，这意味着将一个线程投入到每个用户，这可能非常昂贵。

为了最小化并发连接的成本，Tornado使用单线程事件循环。这意味着所有应用程序代码都应该是异步和非阻塞的，因为一次只能有一个操作处于活动状态。

术语异步和非阻塞是密切相关的，并且通常可以互换使用，但它们并不完全相同。

## 阻塞
函数在返回之前等待某事发生时会阻塞。一个函数可能会出于多种原因而阻塞：网络I/O，磁盘I/O，互斥等。在Tornado的上下文中，我们通常谈论在网络I/O的上下文中阻塞，尽管要最小化所有类型的阻塞。

## 异步
一个异步函数返回完成之前，通常会导致一些工作在后台发生，在应用程序中触发了一些未来的行动（而不是正常的同步功能，做他们会在返回前做的一切）。异步接口有很多种风格：

- Callback argument
- Return a placeholder (Future, Promise, Deferred)
- Deliver to a queue
- Callback registry (e.g. POSIX signals)

Tornado中的异步操作通常返回占位符对象（Futures），但一些低级组件除外，例如IOLoop使用回调。Futures通常会使用await或yield关键字将其转换为结果。

## 示例
这是一个同步函数示例：
```python
from tornado.httpclient import HTTPClient

def synchronous_fetch(url):
    http_client = HTTPClient()
    response = http_client.fetch(url)
    return response.body
```

这里是与本机协程异步重写的相同函数：
```python
from tornado.httpclient import AsyncHTTPClient

async def asynchronous_fetch(url):
    http_client = AsyncHTTPClient()
    response = await http_client.fetch(url)
    return response.body
```

或者为了与旧版本的Python兼容，使用`tornado.gen`模块：
```python
from tornado.httpclient import AsyncHTTPClient
from tornado import gen

@gen.coroutine
def async_fetch_gen(url):
    http_client = AsyncHTTPClient()
    response = yield http_client.fetch(url)
    raise gen.Return(response.body)
```

协同程序有点神奇，但它们在内部做的是这样的：
```python
from tornado.concurrent import Future

def async_fetch_manual(url):
    http_client = AsyncHTTPClient()
    my_future = Future()
    fetch_future = http_client.fetch(url)
    def on_fetch(f):
        my_future.set_result(f.result().body)
    fetch_future.add_done_callback(on_fetch)
    return my_future
```

## 协程
Coroutines是在Tornado中编写异步代码的推荐方法。协同程序使用Python await或yield关键字来挂起和恢复执行而不是一系列回调。原生与装饰协程：
```
# Decorated:                    # Native:

# Normal function declaration
# with decorator                # "async def" keywords
@gen.coroutine
def a():                        async def a():
    # "yield" all async funcs       # "await" all async funcs
    b = yield c()                   b = await c()
    # "return" and "yield"
    # cannot be mixed in
    # Python 2, so raise a
    # special exception.            # Return normally
    raise gen.Return(b)             return b
```

这是coroutine装饰器内循环的简化版本：
```python
# Simplified inner loop of tornado.gen.Runner
def run(self):
    # send(x) makes the current yield return x.
    # It returns when the next yield is reached
    future = self.gen.send(self.next)
    def callback(f):
        self.next = f.result()
        self.run()
    future.add_done_callback(callback)
```

## 参考资料：
- [Asynchronous](http://www.tornadoweb.org/en/stable/guide/async.html)
- [coroutines](http://www.tornadoweb.org/en/stable/guide/coroutines.html)
- [concurrent](http://www.tornadoweb.org/en/stable/concurrent.html)