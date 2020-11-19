title: 深入理解 Python 异步编程
date: 2017-08-27
tags: [Python]
---
如今,地球上最发达,规模最庞大的计算机程序,莫过于因特网.而从CPU的时间观中可知,网络I/O是最大的I/O瓶颈,除了宕机没有比它更慢的.所以,诸多异步框架都对准的是网络I/O.

<!--more-->
## 一些概念

### 阻塞
程序未得到所需计算资源时被挂起的状态.程序在等待某个操作完成期间,自身无法继续干别的事情,则称该程序在该操作上是阻塞的.

### 非阻塞
程序在等待某操作过程中,自身不被阻塞,可以继续运行干别的事情,则称该程序在该操作上是非阻塞的.

### 异步
为完成某个任务,不同程序单元之间过程中无需通信协调,也能完成任务的方式.简言之,异步意味着无序.

### 并发
并发描述的是程序的组织结构.指程序要被设计成多个可独立执行的子任务,以利用有限的计算机资源使多个任务可以被实时或近实时执行为目的.

### 并行
并行描述的是程序的执行状态.指多个任务同时被执行,以利用富余计算资源(多核CPU)加速完成多个任务为目的.

### 小结

- 并行是为了利用多核加速多任务完成的进度
- 并发是为了让独立的子任务都有机会被尽快执行,但不一定能加速整体进度
- 非阻塞是为了提高程序整体执行效率
- 异步是高效地组织非阻塞任务的方式

要支持并发,必须拆分为多任务,不同任务相对而言才有阻塞/非阻塞,同步/异步.所以,并发,异步,非阻塞三个词总是如影随形.

## 简易的服务
先建个简易的异步服务:
```
#coding=utf-8
import tornado.ioloop
import tornado.web
from tornado import gen

class Index(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        yield gen.sleep(1)
        data = self.get_argument('data')
        self.write(' get:' + data)

handlers = [(r'/', Index)]
app = tornado.web.Application(handlers)

if __name__ == '__main__':
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

## 同步阻塞方式
```
import socket

def blocking_way():
    sock = socket.socket()
    sock.connect(('127.0.0.1', 9000))
    url = '/?data=test'
    host = '127.0.0.1:9000'
    get_str = 'GET %s HTTP/1.0\r\nHost: %s\r\n\r\n' % (url, host)
    sock.send(get_str.encode('utf8'))
    response = b''
    chunk = sock.recv(4096)
    while chunk:
        response += chunk
        chunk = sock.recv(4096)
    return response

def sync_way():
    res = []
    for i in range(10):
        res.append(blocking_way())
    return res

import time
ts = time.time()
sync_way()
print('time loss: {0:.4f}'.format(time.time() - ts))
```

`time loss: 10.0421`.在示例代码中有两个关键点:一是第5行的`sock.connect(('127.0.0.1', 9000))`,该调用的作用是向本机的9000端口发起网络连接请求;二是第11行,第14行的`sock.recv(4096)`,该调用的作用是从socket上读取4K字节数据.

我们知道,创建网络连接,多久能创建完成不是客户端决定的,而是由网络状况和服务端处理能力共同决定.服务端什么时候返回了响应数据并被客户端接收到可供程序读取,也是不可预测的.所以,`sock.connect()`和`sock.recv()`这两个调用在默认情况下是阻塞的.

## 改进: 多线程
```
import socket

def blocking_way():
    sock = socket.socket()
    sock.connect(('127.0.0.1', 9000))
    url = '/?data=test'
    host = '127.0.0.1:9000'
    get_str = 'GET %s HTTP/1.0\r\nHost: %s\r\n\r\n' % (url, host)
    sock.send(get_str.encode('utf8'))
    response = b''
    chunk = sock.recv(4096)
    while chunk:
        response += chunk
        chunk = sock.recv(4096)
    return response

from concurrent.futures import ThreadPoolExecutor
def process_way():
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = [executor.submit(blocking_way) for i in range(10)]
    return [result.result() for result in results]

import time
ts = time.time()
process_way()
print('time loss: {0:.4f}'.format(time.time() - ts))
```

`time loss: 1.0186`.但是,多线程仍有问题,特别是Python里的多线程.首先,Python中的多线程因为GIL的存在,它们并不能利用CPU多核优势,一个Python进程中,只允许有一个线程处于运行状态.除了GIL之外,所有的多线程还有通病,它们是被OS调度,调度策略是抢占式的,以保证同等优先级的线程都有均等的执行机会,所以就存在竞态条件.

而且线程支持的多任务规模,在数百到数千的数量规模.在大规模的高频网络交互系统中,仍然有些吃力.当然,多线程最主要的问题还是竞态条件.

## 非阻塞方式
终于,我们来到了非阻塞解决方案.先来看看最原始的非阻塞如何工作的:
```
import socket

def noblocking_way():
    sock = socket.socket()
    sock.setblocking(False)
    try:
        sock.connect(('127.0.0.1', 9000))
    except Exception:
        pass
    url = '/?data=test'
    host = '127.0.0.1:9000'
    get_str = 'GET %s HTTP/1.0\r\nHost: %s\r\n\r\n' % (url, host)
    while True:
        try:
            sock.send(get_str.encode('utf8'))
            break
        except Exception:
            pass
    response = b''
    while True:
        try:
            chunk = sock.recv(4096)
            while chunk:
                response += chunk
                chunk = sock.recv(4096)
            break
        except Exception:
            pass
    return response

def sync_way():
    res = []
    for i in range(10):
        res.append(noblocking_way())
    return res

import time
ts = time.time()
sync_way()
print('time loss: {0:.4f}'.format(time.time() - ts))
```

`time loss: 10.0173`.首先注意到两点,就感觉被骗了.一是耗时与同步阻塞相当,二是代码更复杂.要非阻塞何用?

`sock.setblocking(False)`告诉OS,让socket上阻塞调用都改为非阻塞的方式.上述代码在执行完`sock.connect()`和`sock.recv()`后的确不再阻塞,可以继续往下执行请求准备的代码或者是执行下一次读取.

代码变得更复杂也是上述原因所致.第7行要放在try语句内,是因为socket在发送非阻塞连接请求过程中,系统底层也会抛出异常.`connect()`被调用之后,立即可以往下执行第10和11行的代码.

需要while循环不断尝试`send()`,是因为`connect()`已经非阻塞,在`send()`之时并不知道socket的连接是否就绪,只有不断尝试,尝试成功为止,即发送数据成功了.`recv()`调用也是同理.

虽然`connect()`和`recv()`不再阻塞主程序,空出来的时间段CPU没有空闲着,但并没有利用好这空闲去做其他有意义的事情,而是在循环尝试读写socket,不停判断非阻塞调用的状态是否就绪.还得处理来自底层的可忽略的异常.

然后10次下载任务仍然按序进行,所以总体执行时间和同步阻塞相当.如果非得这样子,那还不如同步阻塞算了.

## 改进: 使用回调

### Epoll
判断非阻塞调用是否就绪如果OS能做,是不是应用程序就可以不用自己去等待和判断了,就可以利用这个空闲去做其他事情以提高效率.

所以OS将I/O状态的变化都封装成了事件,如可读事件/可写事件.并且提供了专门的系统模块让应用程序可以接收事件通知,这个模块就是select.让应用程序可以通过select注册文件描述符和回调函数.当文件描述符的状态发生变化时,select就调用事先注册的回调函数.

select因其算法效率比较低,后来改进成了poll.再后来又有进一步改进,BSD内核改进成了kqueue模块,而Linux内核改进成了epoll模块.这四个模块的作用都相同,暴露给程序员使用的API也几乎一致,区别在于kqueue和epoll在处理大量文件描述符时效率更高.

鉴于Linux服务器的普遍性,以及为了追求更高效率,所以我们常常听闻被探讨的模块都是epoll.

### Callback
把I/O事件的等待和监听任务交给了OS,那OS在知道I/O状态发生改变后,例如socket连接已建立成功可发送数据,它又怎么知道接下来该干嘛呢?只能回调.

需要我们将发送数据与读取数据封装成独立的函数,让epoll代替应用程序监听socket状态时,得告诉epoll:如果socket状态变为可以往里写数据,连接建立成功了,请调用HTTP请求发送函数;如果socket变为可以读数据了,客户端已收到响应,请调用响应处理函数.

于是我们利用epoll结合回调机制重构爬虫代码:
```
import socket
from selectors import DefaultSelector, EVENT_WRITE, EVENT_READ

selector = DefaultSelector()
stopped = False
urls = {'/?data=%d'% i for i in range(10)}

class Crawler(object):
    def __init__(self, url):
        self.url = url
        self.sock = None
        self.response = b''

    def fetch(self):
        self.sock = socket.socket()
        self.sock.setblocking(False)
        try:
            self.sock.connect(('127.0.0.1', 9000))
        except Exception:
            pass
        selector.register(self.sock.fileno(), EVENT_WRITE, self.connected)

    def connected(self, key, mask):
        selector.unregister(key.fd)
        get_str = 'GET %s HTTP/1.0\r\nHost: %s\r\n\r\n' % (self.url, '127.0.0.1:9000')
        self.sock.send(get_str.encode('utf8'))
        selector.register(key.fd, EVENT_READ, self.read_response)

    def read_response(self, key, mask):
        global stopped
        chunk = self.sock.recv(4096)
        if chunk:
            self.response += chunk
        else:
            selector.unregister(key.fd)
            urls.remove(self.url)
            if len(urls) < 1:
                stopped = True
```

此处和前面稍有不同的是,我们将下载不同的10个页面,相对URL路径存放于urls列表中.现在看看改进:

- 首先,不断尝试`send()`和`recv()`的两个循环被消灭掉了
- 其次,导入了selectors模块,并创建了一个DefaultSelector实例
- 然后,在第21行和第27行分别注册了socket可写事件和可读事件发生后的回调函数

### Event Loop
虽然代码结构清晰了,阻塞操作也交给OS去等待和通知了.但是,我们要抓取10个不同页面,就得创建10个Crawler实例,就有20个事件将要发生,那如何从selector里获取当前正发生的事件,并且得到对应的回调函数去执行呢?

我们只得采用老办法,写一个循环,去访问selector模块,等待它告诉我们当前是哪个事件发生了,应该对应哪个回调.这个等待事件通知的循环,称之为事件循环:
```
def loop():
    while not stopped:
        events = selector.select()
        for event_key, event_mask in events:
            callback = event_key.data
            callback(event_key, event_mask)
```

上述代码中,我们用stopped全局变量控制事件循环何时停止.当urls消耗完毕后,会标记stopped为True.

重要的是第3行代码`selector.select()`是一个阻塞调用,因为如果事件不发生,那应用程序就没事件可处理,所以就干脆阻塞在这里等待事件发生.那可以推断,如果只下载一篇网页,一定要`connect()`之后才能`send()`继而`recv()`,那它的效率和阻塞的方式是一样的.因为不在`connect()/recv()`上阻塞,也得在`select()`上阻塞.

所以,selector机制是设计用来解决大量并发连接的.当系统中有大量非阻塞调用,能随时产生事件的时候,selector机制才能发挥最大的威力.

下面创建10个下载任务和启动事件循环的:
```
if __name__ == '__main__':
    import time
    ts = time.time()
    for url in urls:
        crawler = Crawler(url)
        crawler.fetch()
    loop()
    print('time loss: {0:.4f}'.format(time.time() - ts))
```

`time loss: 1.0174`,结果令人振奋.在单线程内用`事件循环+回调`搞定了10篇网页同时下载的问题.这,已经是异步编程了,虽然有一个for循环顺序地创建Crawler实例并调用fetch方法,但是fetch内仅有`connect()`和注册可写事件,而且从执行时间明显可以推断,多个下载任务确实在同时进行!

上述代码异步执行的过程:

- 创建Crawler实例
- 调用fetch方法,会创建socket连接和在selector上注册可写事件
- fetch内并无阻塞操作,该方法立即返回
- 重复上述3个步骤,将10个不同的下载任务都加入事件循环
- 启动事件循环,进入第1轮循环,阻塞在事件监听上
- 当某个下载任务`EVENT_WRITE`被触发,回调其connected方法,第一轮事件循环结束
- 进入第2轮事件循环,当某个下载任务有事件触发,执行其回调函数.此时已经不能推测是哪个事件发生,因为有可能是上次connected里的`EVENT_READ`先被触发,也可能是其他某个任务的`EVENT_WRITE`被触发
- 循环往复,直至所有下载任务被处理完成
- 退出事件循环,结束整个下载程序

### 小结
目前为止,我们已经从同步阻塞学习到了异步非阻塞.掌握了在单线程内同时并发执行多个网络I/O阻塞型任务的黑魔法.而且与多线程相比,连线程切换都没有了,执行回调函数是函数调用开销,在线程的栈内完成,因此性能也更好.单机支持的任务规模也变成了数万到数十万个.

## 改进: 使用协程
协程(Co-routine),即是协作式的例程.它是非抢占式的多任务子例程的概括,可以允许有多个入口点在例程中确定的位置来控制程序的暂停与恢复执行.

例程是什么?编程语言定义的可被调用的代码段,为了完成某个特定功能而封装在一起的一系列指令.一般的编程语言都用称为函数或方法的代码结构来体现.

### 生成器与协程
Python中有种特殊的对象,Generator,它的特点和协程很像.每次迭代之间,会暂停执行,继续下一次迭代的时候还不会丢失先前的状态.

为了支持用生成器做简单的协程,Python对生成器进行了增强,生成器可以通过yield暂停执行和向外返回数据,也可以通过`send()`向生成器内发送数据,还可以通过`throw()`向生成器内抛出异常以便随时终止生成器的运行.

### 未来对象(Future)
不用回调的方式,怎么知道异步调用的结果呢?先设计一个对象,异步调用执行完的时候,就把结果放在它里面.这种对象称之为未来对象.
```
import socket
from selectors import DefaultSelector, EVENT_WRITE, EVENT_READ

selector = DefaultSelector()
stopped = False
urls = {'/?data=%d'% i for i in range(10)}

class Future(object):
    def __init__(self):
        self.result = None
        self._callbacks = []

    def add_done_callback(self, fn):
        self._callbacks.append(fn)

    def set_result(self, result):
        self.result = result
        for fn in self._callbacks:
            fn(self)
```

未来对象有一个result属性,用于存放未来的执行结果.还有个`set_result()`方法,是用于设置result的,并且会在给result绑定值以后运行事先给future添加的回调.回调是通过未来对象的`add_done_callback()`方法添加的.

### 重构Crawler
```
class Crawler(object):
    def __init__(self, url):
        self.url = url
        self.response = b''

    def fetch(self):
        sock = socket.socket()
        sock.setblocking(False)
        try:
            sock.connect(('127.0.0.1', 9000))
        except Exception:
            pass
        f = Future()

        def on_connected():
            f.set_result(None)

        selector.register(sock.fileno(), EVENT_WRITE, on_connected)
        yield f
        selector.unregister(sock.fileno())
        get_str = 'GET %s HTTP/1.0\r\nHost: %s\r\n\r\n' % (self.url, '127.0.0.1:9000')
        sock.send(get_str.encode('utf8'))

        global stopped
        while True:
            f = Future()

            def on_readable():
                f.set_result(sock.recv(4096))

            selector.register(sock.fileno(), EVENT_READ, on_readable)
            chunk = yield f
            selector.unregister(sock.fileno())
            if chunk:
                self.response += chunk
            else:
                urls.remove(self.url)
                if len(urls) < 1:
                    stopped = True
                break
```

和先前的回调版本对比,已经有了较大差异.fetch方法内有了yield表达式,使它成为了生成器.我们知道生成器需要先调用`next()`迭代一次或者是先`send(None)`启动,遇到yield之后便暂停.那这fetch生成器如何再次恢复执行呢?至少Future和Crawler都没看到相关代码.

### 任务对象(Task)
```
class Task(object):
    def __init__(self, coro):
        self.coro = coro
        f = Future()
        f.set_result(None)
        self.step(f)

    def step(self, future):
        try:
            next_future = self.coro.send(future.result)
        except Exception:
            return
        next_future.add_done_callback(self.step)
```

Task封装了coro对象,即初始化时传递给他的对象,被管理的任务是待执行的协程,故而这里的coro就是`fetch()`生成器.它还有个`step()`方法,在初始化的时候就会执行一遍.`step()`内会调用生成器的`send()`方法,初始化第一次发送的是None就驱动了coro即`fetch()`的第一次执行.

`send()`完成之后,得到下一次的future,然后给下一次的future添加`step()`回调.原来`add_done_callback()`不是给写爬虫业务逻辑用的,此前的callback干的是业务逻辑.

### 事件循环(Event Loop)
```
def loop():
    while not stopped:
        events = selector.select()
        for event_key, event_mask in events:
            callback = event_key.data
            callback()

if __name__ == '__main__':
    import time
    ts = time.time()
    for url in urls:
        crawler = Crawler(url)
        Task(crawler.fetch())
    loop()
    print('time loss: {0:.4f}'.format(time.time() - ts))
```

`time loss: 1.0188`.现在loop有了些许变化,`callback()`不再传递`event_key`和`event_mask`参数.也就是说,这里的回调根本不关心是谁触发了这个事件,结合`fetch()`可以知道,它只需完成对future设置结果值即可`f.set_result()`.而且future是谁它也不关心,因为协程能够保存自己的状态,知道自己的future是哪个.也不用关心到底要设置什么值,因为要设置什么值也是协程内安排的.

## 改进: 使用 yield from
`yield from`是py33新引入的语法.它主要解决的就是在生成器里玩生成器不方便的问题.

### 重构代码
抽象socket连接的功能:
```
import socket
from selectors import DefaultSelector, EVENT_WRITE, EVENT_READ

selector = DefaultSelector()
stopped = False
urls = {'/?data=%d'% i for i in range(10)}

def connect(sock, address):
    f = Future()
    sock.setblocking(False)
    try:
        sock.connect(address)
    except Exception:
        pass

    def on_connected():
        f.set_result(None)

    selector.register(sock.fileno(), EVENT_WRITE, on_connected)
    yield from f
    selector.unregister(sock.fileno())
```

抽象单次`recv()`和读取完整的response功能:
```
def read(sock):
    f = Future()

    def on_readable():
        f.set_result(sock.recv(4096))

    selector.register(sock.fileno(), EVENT_READ, on_readable)
    chunk = yield from f
    selector.unregister(sock.fileno())
    return chunk

def read_all(sock):
    response = []
    chunk = yield from read(sock)
    while chunk:
        response.append(chunk)
        chunk = yield from read(sock)
    return b''.join(response)
```

三个关键点的抽象已经完成,现在重构Crawler类:
```
class Crawler(object):
    def __init__(self, url):
        self.url = url
        self.response = b''

    def fetch(self):
        global stopped
        sock = socket.socket()
        yield from connect(sock, ('127.0.0.1', 9000))
        get_str = 'GET %s HTTP/1.0\r\nHost: %s\r\n\r\n' % (self.url, '127.0.0.1:9000')
        sock.send(get_str.encode('utf8'))
        self.response = yield from read_all(sock)
        urls.remove(self.url)
        if len(urls) < 1:
            stopped = True
```

上面代码整体来讲没什么问题,可复用的代码已经抽象出去,作为子生成器也可以使用`yield from`语法来获取值.yield可以直接作用于普通Python对象,而`yield from`却不行,所以我们对Future还要进一步改造,把它变成一个iterable对象:
```
class Future(object):
    def __init__(self):
        self.result = None
        self._callbacks = []

    def add_done_callback(self, fn):
        self._callbacks.append(fn)

    def set_result(self, result):
        self.result = result
        for fn in self._callbacks:
            fn(self)

    def __iter__(self):
        yield self
        return self.result
```

只是增加了`__iter__()`方法的实现.如果不把Future改成iterable也是可以的,还是用原来的`yield f`即可.那为什么需要改进呢?

- 首先,我们是在基于生成器做协程,而生成器还得是生成器,如果继续混用yield和`yield from`做协程,代码可读性和可理解性都不好
- 其次,如果不改,协程内还得关心它等待的对象是否可被yield,如果协程里还想继续返回协程怎么办?如果想调用普通函数动态生成一个Future对象再返回怎么办?

所以,在py33引入`yield from`新语法之后,就不再推荐用yield去做协程.全都使用`yield from`,由于其双向通道的功能,可以让我们在协程间随心所欲地传递数据.

```
class Task(object):
    def __init__(self, coro):
        self.coro = coro
        f = Future()
        f.set_result(None)
        self.step(f)

    def step(self, future):
        try:
            next_future = self.coro.send(future.result)
        except Exception:
            return
        next_future.add_done_callback(self.step)

def loop():
    while not stopped:
        events = selector.select()
        for event_key, event_mask in events:
            callback = event_key.data
            callback()

if __name__ == '__main__':
    import time
    ts = time.time()
    for url in urls:
        crawler = Crawler(url)
        Task(crawler.fetch())
    loop()
    print('time loss: {0:.4f}'.format(time.time() - ts))
```

Task和loop不变,`time loss: 1.0186`.用`yield from`改进基于生成器的协程,代码抽象程度更高.使业务逻辑相关的代码更精简.由于其双向通道功能可以让协程之间随心所欲传递数据,使Python异步编程的协程解决方案大大向前迈进了一步.

## asyncio和原生协程初体验
asyncio是py34试验性引入的异步I/O框架,提供了基于协程做异步I/O编写单线程并发代码的基础设施.其核心组件有事件循环(Event Loop),协程(Coroutine),任务(Task),未来对象(Future)以及其他一些扩充和辅助性质的模块.

实际上,真正的asyncio要复杂得多,它还实现了零拷贝,公平调度,异常处理,任务状态管理等等使Python异步编程更完善的内容.

在py35中新增了async/await语法,对协程有了明确而显式的支持,称之为原生协程.async/await和`yield from`这两种风格的协程底层复用共同的实现,而且相互兼容.

```
import asyncio
import aiohttp

host = 'http://127.0.0.1:9000'
urls = {'/?data=%d'% i for i in range(10)}

loop = asyncio.get_event_loop()

async def fetch(url):
    async with aiohttp.ClientSession(loop=loop) as session:
        async with session.get(url) as response:
            response = await response.read()
            return response

if __name__ == '__main__':
    import time
    ts = time.time()
    tasks = [fetch(host+url) for url in urls]
    loop.run_until_complete(asyncio.gather(*tasks))
    print('time loss: {0:.4f}'.format(time.time() - ts))
```

`time loss: 1.0256`,运行前可能需要`pip install aiohttp`.使用asyncio库后变化很大:

- 没有了yield或`yield from`,而是async/await
- 没有了自造的`loop()`,取而代之的是`asyncio.get_event_loop()`
- 无需自己在socket上做异步操作,不用显式地注册和注销事件,aiohttp库已经代劳
- 没有了显式的Future和Task,asyncio已封装
- 更少量的代码,更优雅的设计

## 参考资料:
- [深入理解 Python 异步编程 上](http://python.jobbole.com/88291/)