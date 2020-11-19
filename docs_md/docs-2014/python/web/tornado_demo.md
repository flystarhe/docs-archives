# Tornado Demo

## Hello, world
```python
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

def make_app():
    return tornado.web.Application([(r"/", MainHandler),])

if __name__ == "__main__":
    make_app().listen(8888)
    tornado.ioloop.IOLoop.current().start()
```

## Coroutines
synchronous function:
```python
from tornado.httpclient import HTTPClient

def synchronous_fetch(url):
    http_client = HTTPClient()
    response = http_client.fetch(url)
    return response.body
```

be asynchronous:
```python
from tornado.concurrent import Future
from tornado.httpclient import AsyncHTTPClient

def async_fetch_future(url):
    http_client = AsyncHTTPClient()
    my_future = Future()
    fetch_future = http_client.fetch(url)
    fetch_future.add_done_callback(lambda f: my_future.set_result(f.result()))
    return my_future
```

recommended asynchronous:
```python
import tornado.gen

@tornado.gen.coroutine
def fetch_coroutine(url):
    http_client = AsyncHTTPClient()
    response = yield http_client.fetch(url)
    return response.body
```

`async` and `await` will run faster:
```python
async def fetch_coroutine(url):
    http_client = AsyncHTTPClient()
    response = await http_client.fetch(url)
    return response.body
```

## Coroutine patterns
a blocking function from a coroutine is to use `IOLoop.run_in_executor`:
```python
from tornado.ioloop import IOLoop
from tornado.concurrent import futures
executor = futures.ThreadPoolExecutor(6)
@tornado.gen.coroutine
def call_blocking():
    yield IOLoop.current().run_in_executor(executor, blocking_func, args)
```

Parallelism:
```python
@tornado.gen.coroutine
def parallel_fetch(url1, url2):
    resp1, resp2 = yield [http_client.fetch(url1),
                          http_client.fetch(url2)]

@tornado.gen.coroutine
def parallel_fetch_many(urls):
    responses = yield [http_client.fetch(url) for url in urls]
    # responses is a list of HTTPResponses in the same order

@tornado.gen.coroutine
def parallel_fetch_dict(urls):
    responses = yield {url: http_client.fetch(url) for url in urls}
    # responses is a dict {url: HTTPResponse}
```

Interleaving:
```python
@tornado.gen.coroutine
def get(self):
    fetch_future = self.fetch_next_chunk()
    while True:
        chunk = yield fetch_future
        if chunk is None:
            break
        self.write(chunk)
        fetch_future = self.fetch_next_chunk()
        yield self.flush()
```

## Asynchronous handlers
```python
import base64
import traceback
import tornado.gen
import tornado.websocket
from tornado.concurrent import futures

_threads = futures.ThreadPoolExecutor(6)

def on_message_worker(message):
    try:
        if not isinstance(message, bytes):
            message = base64.b64decode(message)
        result = "{} bytes".format(len(message))
    except Exception:
        result = "-1,{}".format(traceback.format_exc())
    return result

class WebSocketSearch(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    @tornado.gen.coroutine
    def on_message(self, message):
        result = yield _threads.submit(on_message_worker, message)
        yield self.write_message(result)
```

## Running and deploying
```python
import tornado.gen
import tornado.web
import tornado.ioloop
handlers = ..
tornado.web.Application(handlers).listen(8888)
tornado.ioloop.IOLoop.current().start()
```

multiple processes:[limitations](http://www.tornadoweb.org/en/stable/guide/running.html#processes-and-ports)
```python
import tornado.web
import tornado.ioloop
handlers = ..
app = tornado.web.Application(handlers)
server = HTTPServer(app)
server.bind(8888)
server.start(0)
tornado.ioloop.IOLoop.current().start()
```

## https(ssl)
Tornado本身支持SSL,所以我们这里需要做的主要是生成可用的证书.

### 生成证书
CSR文件必须有CA的签名才可形成证书,先自己做CA:
```bash
$ mkdir -p demoCA/newcerts
$ touch demoCA/index.txt
$ echo "0000" >> demoCA/serial
$ openssl req -new -x509 -keyout ca.key -out ca.crt
```

生成服务器端key:
```bash
$ openssl genrsa -des3 -out server.key 1024
```

生成服务器待签CSR文件:
```bash
$ openssl req -new -key server.key -out server.csr
```

用生成的CA的证书为刚才生成的`server.csr,client.csr`文件签名:
```bash
$ openssl ca -in server.csr -out server.crt -cert ca.crt -keyfile ca.key
```

### 使用证书
```python
#app.py
#python app.py
import tornado.web
import tornado.ioloop


class Info(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world!")
        self.finish()


import ssl
ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_ctx.load_cert_chain("ssl/server.crt", "ssl/server.key")
if __name__ == "__main__":
    handlers = [(r"/", Info)]
    tornado.web.Application(handlers).listen(8888, ssl_options=ssl_ctx)
    tornado.ioloop.IOLoop.current().start()
```
