title: 使用Tornado提供web服务
date: 2017-08-08
tags: [Python]
---
Tornado和现在的主流Web服务器框架,和大多数Python框架有着明显的区别:它是非阻塞式服务器,而且速度很快.也是比较常用的Python开源框架之一.

<!--more-->
## 安装Tornado
```
pip install tornado
```

## 搭建服务
使用Tornado搭建web服务是容易的,`hej.py`:
```
#coding=utf-8
import tornado.ioloop
import tornado.web

class Index(tornado.web.RequestHandler):
    def get(self):
        data = self.get_argument('data')
        self.write(' get:' + data)

class Guess(tornado.web.RequestHandler):
    def post(self):
        data = self.get_argument('data')
        self.write('post:' + data)

router = [(r'/', Index),
          (r'/guess', Guess)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

## 启动服务
命令行输入`python hej.py`.

## 测试服务
测试GET方法:
```
import requests

url = 'http://127.0.0.1:9000/'
vals = {'data': '[1,2,3]'}

response = requests.get(url, params=vals)
print(response.status_code)
print(response.url)
print(response.text)
```

测试POST方法:
```
import requests

url = 'http://127.0.0.1:9000/guess'
vals = {'data': '[1,2,3]'}

response = requests.post(url, data=vals)
print(response.status_code)
print(response.url)
print(response.text)
```

## 阻塞的服务
这里用`time.sleep`模拟耗时任务,代码如下:
```
import tornado.ioloop
import tornado.web
import time

class Index(tornado.web.RequestHandler):
    def get(self):
        ts = time.strftime('%H:%M:%S')
        te = self.doing(5)
        self.write('Index:ts={0},te={1}'.format(ts, te))

    def doing(self, n):
        time.sleep(n)
        te = time.strftime('%H:%M:%S')
        return te

class Guess(tornado.web.RequestHandler):
    def get(self):
        ts = time.strftime('%H:%M:%S')
        te = time.strftime('%H:%M:%S')
        self.write('Guess:ts={0},te={1}'.format(ts, te))

router = [(r'/', Index),
          (r'/guess', Guess)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

打开两个终端,先终端1执行`curl 127.0.0.1:9000`,几乎同时终端2执行`curl 127.0.0.1:9000/guess`,终端1输出`Index:ts=08:30:10,te=08:30:15`,终端2输出`Guess:ts=08:30:15,te=08:30:15`.显而易见,终端2的请求是服务器完成终端1的请求后才开始处理的.

## 非阻塞的服务
正式开始之前要知道如何编写Tornado中的异步函数:
```
    @gen.coroutine
    def doing(self, n):
        yield gen.sleep(n)
        te = time.strftime('%H:%M:%S')
        return te
```

使用`coroutine`方式有个很明显的缺点,就是严重依赖第三方的实现,如果库本身不支持Tornado异步操作,使用了协程依然会是阻塞的.比如,使用`yield gen.sleep(n)`是非阻塞的,使用`time.sleep(n)`则依然阻塞.非阻塞版代码如下:
```
import tornado.ioloop
import tornado.web
import time

from tornado import gen

class Index(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        ts = time.strftime('%H:%M:%S')
        te = yield self.doing(5)
        self.write('Index:ts={0},te={1}'.format(ts, te))

    @gen.coroutine
    def doing(self, n):
        yield gen.sleep(n)
        te = time.strftime('%H:%M:%S')
        return te

class Guess(tornado.web.RequestHandler):
    def get(self):
        ts = time.strftime('%H:%M:%S')
        te = time.strftime('%H:%M:%S')
        self.write('Guess:ts={0},te={1}'.format(ts, te))

router = [(r'/', Index),
          (r'/guess', Guess)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

这种异步非阻塞的方式需要依赖大量的基于Tornado协议的异步库,使用上会比较受限.

## 耗时任务与多线程
```
import tornado.ioloop
import tornado.web
import time

from tornado import gen
from tornado.concurrent import futures

class Index(tornado.web.RequestHandler):
    thread_pool = futures.ThreadPoolExecutor(4)

    @gen.coroutine
    def get(self):
        ts = time.strftime('%H:%M:%S')
        te = yield self.thread_pool.submit(self.doing, 5)
        self.write('Index:ts={0},te={1}'.format(ts, te))

    def doing(self, n):
        time.sleep(n)
        te = time.strftime('%H:%M:%S')
        return te

class Guess(tornado.web.RequestHandler):
    def get(self):
        ts = time.strftime('%H:%M:%S')
        te = time.strftime('%H:%M:%S')
        self.write('Guess:ts={0},te={1}'.format(ts, te))

router = [(r'/', Index),
          (r'/guess', Guess)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

打开三个终端,先终端1执行`curl 127.0.0.1:9000`,几乎同时终端2执行`curl 127.0.0.1:9000`,几乎同时终端3执行`curl 127.0.0.1:9000/guess`,终端1输出`Index:ts=10:48:12,te=10:48:17`,终端2输出`Index:ts=10:48:12,te=10:48:17`,终端3输出`Guess:ts=10:48:13,te=10:48:13`.很好,不再阻塞了.

### 优雅的实现
```
import tornado.ioloop
import tornado.web
import time

from tornado import gen
from tornado.concurrent import futures, run_on_executor

class Base(tornado.web.RequestHandler):
    thread_pool = futures.ThreadPoolExecutor(4)

class Index(Base):
    @gen.coroutine
    def get(self):
        ts = time.strftime('%H:%M:%S')
        te = yield self.doing(5)
        self.write('Index:ts={0},te={1}'.format(ts, te))

    @run_on_executor(executor='thread_pool')
    def doing(self, n):
        time.sleep(n)
        te = time.strftime('%H:%M:%S')
        return te

class Guess(Base):
    @run_on_executor(executor='thread_pool')
    def get(self):
        ts = time.strftime('%H:%M:%S')
        time.sleep(3)
        te = time.strftime('%H:%M:%S')
        self.write('Guess:ts={0},te={1}'.format(ts, te))

router = [(r'/', Index),
          (r'/guess', Guess)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

## 模板
建议监听端口由命令行参数提供,而不是固定在代码中,比如用`sys.argv[1]`,或`argparse`库.对于异常使用`raise HTTPError(400, 'log message')`也是推荐的.
```
################################################################
# ./src/web/base.py
################################################################
#coding=utf-8
import logging
import logging.handlers as handlers
filehandler = handlers.RotatingFileHandler('log.web', maxBytes=10**6, backupCount=7)
filehandler.setLevel(logging.INFO)
filehandler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s:%(message)s'))
logger = logging.getLogger('web')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)

import tornado.web
from tornado import gen
from tornado.concurrent import futures
class Base(tornado.web.RequestHandler):
    thread_pool = futures.ThreadPoolExecutor(2)

    def options(self):
        #raise HTTPError(405)  #允许跨域
        self.set_status(200)
        self.finish()

    @gen.coroutine
    def get(self):
        res, log = yield self.worker()
        self.write(res)
        self.finish()
        logger.info(log)

    @gen.coroutine
    def post(self):
        res, log = yield self.worker()
        self.write(res)
        self.finish()
        logger.info(log)

################################################################
# ./src/web/worker.py
################################################################
import json
import traceback
from .base import Base
from tornado.web import HTTPError
from tornado.concurrent import run_on_executor
class Guess(Base):
    @run_on_executor(executor='thread_pool')
    def worker(self):
        arguments = self.request.arguments
        try:
            var = arguments['name'][0].decode('utf8')
            # computing task
            msg = str(var)
            # result
            res = json.dumps({'state': 0, 'result': msg})
            log = 'name:{}'.format(var)
        except Exception:
            raise HTTPError(400, traceback.format_exc())
        return res, log

################################################################
# ./server.py
################################################################
import sys
import tornado.ioloop
import tornado.web
from src.web.worker import Guess

router = [(r'/', Guess),]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(sys.argv[1])
    tornado.ioloop.IOLoop.current().start()
```

## The Application object
`Application`对象负责全局配置,包括将请求映射到处理程序的路由表.路由表是一个`URLSpec`对象(或元组)的列表,每个对象都包含(至少)一个正则表达式和一个处理程序类.顺序很重要,第一个匹配将被使用.如果正则表达式包含捕获组,则这些组是路径参数,并将传递给处理程序的HTTP方法;如果字典作为第三个元素传递`URLSpec`,它将提供将传`RequestHandler.initialize`初始化参数;最后,`URLSpec`可能会有一个名称,这将允许使用它`RequestHandler.reverse_url`.

例如,在此片段中,将`/`映射到`MainHandler`,并将`/story/`后跟一个数字映射到`StoryHandler`.该号码被传递(作为字符串)给`StoryHandler.get`.
```python
class MainHandler(RequestHandler):
    def get(self):
        self.write('<a href="%s">link to story 1</a>' % self.reverse_url('story', '1'))

class StoryHandler(RequestHandler):
    def initialize(self, db):
        self.db = db

    def get(self, story_id):
        self.write('this is story %s' % story_id)

app = Application(
    [
        url(r'/', MainHandler),
        url(r'/story/([0-9]+)', StoryHandler, dict(db=db), name='story')
    ])
```

## tornado.web.StaticFileHandler
一个简单的处理程序,可以从目录中提供静态内容.为了将一个额外的路径映射到这个处理程序的静态数据目录,你可以在你的应用程序中添加一行:
```python
application = web.Application(
    [
        (r'/content/(.*)', web.StaticFileHandler, {'path': '/var/www'}),
    ])
```

处理程序构造函数需要一个`path`参数,该参数指定要提供内容的本地根目录.请注意,正则表达式中的捕获组解析为`path`传递给`get()`方法.

## tornado.web.RedirectHandler
有两种主要的方式可以在Tornado中重定向请求:`RequestHandler.redirect`和`RedirectHandler`.

您可以在`RequestHandler`内使用`self.redirect()`方法重定向.还有一个可选参数`permanent`,可用于指示重定向被认为是永久性的.`permanentis`的默认值`False`,它会生成一个HTTP响应代码,适用于成功请求后重定向用户等内容.

`RedirectHandler`可让您直接在`Application`路由表中配置重定向.例如,要配置单个静态重定向:
```python
app = tornado.web.Application(
    [
        url(r'/app', tornado.web.RedirectHandler, dict(url='http://itunes.apple.com/my-app-id')),
    ])
```

`RedirectHandler`也支持正则表达式替换.以下规则将重定向所有`/pictures/`以前缀开头的请求`/photos/`:
```python
app = tornado.web.Application(
    [
        url(r'/photos/(.*)', MyPhotoHandler),
        url(r'/pictures/(.*)', tornado.web.RedirectHandler, dict(url=r'/photos/{0}')),
    ])
```

## 参考资料:
- [Tornado](http://www.tornadoweb.org/en/stable/)
- [Tornado 概览](http://www.tornadoweb.cn/documentation/)
- [真正的 Tornado 异步非阻塞](https://hexiangyu.me/posts/15)
- [深入 Tornado 中的协程](http://www.cnblogs.com/MnCu8261/p/6560502.html)