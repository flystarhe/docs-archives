title: Python网络编程之TCP
date: 2017-08-25
tags: [Python]
---
迫于无奈,和TCP杠上了,以前用java玩过,不过都是很久以前的事了,今天用Python再玩一遍.

<!--more-->
## tcp server
使用Tornado的tcpserver模块实现tcpserver:
```python
#coding=utf-8
from tornado.ioloop import IOLoop
from tornado.tcpserver import TCPServer
from tornado.iostream import StreamClosedError
from tornado import gen

class EchoServer(TCPServer):
    @gen.coroutine
    def handle_stream(self, stream, address):
        while True:
            try:
                data = yield stream.read_until(b'\r\n\r\n')
                temp = [line.decode('utf8') for line in data.split(b'\r\n')]
                temp = '\\r\\n'.join(temp)
                temp = temp.encode('utf8')
                yield stream.write(temp+b'\r\n\r\n')
            except StreamClosedError:
                break

if __name__ == '__main__':
    server = EchoServer()
    server.listen(8888)
    IOLoop.current().start()
```

注意,我约定了通信过程中以`\r\n\r\n`作为每轮对话的结束符.

## tcp client (TCPClient)
使用Tornado的tcpclient模块实现tcpclient:
```python
#coding=utf-8
from tornado.ioloop import IOLoop
from tornado.tcpclient import TCPClient
from tornado import gen

@gen.coroutine
def send_message():
    host = '127.0.0.1'
    port = 8888
    message = 'hello world!\r\nhello all?'
    stream = yield TCPClient().connect(host, port)
    yield stream.write((message+'\r\n\r\n').encode('utf8'))
    print('> send:', message, sep='\n')
    result = yield stream.read_until(b'\r\n\r\n')
    result = result.decode('utf8')
    print('> read:', result, sep='\n')

if __name__ == '__main__':
    IOLoop.current().run_sync(send_message)
```

输出为:
```
> send:
hello world!
hello all?
> read:
hello world!\r\nhello all?\r\n\r\n
```

## tcp client (socket)
使用Python的socket模块实现tcpclient:
```python
#coding=utf-8
import socket

class EchoClient(object):
    END = '\r\n\r\n'

    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.cache = ''

    def send(self, text):
        if not text.endswith(self.END):
            text += self.END
        self.sock.send(text.encode('utf8'))

    def recv(self, timeout=None):
        text = ''
        temp = self.cache
        while True:
            temp = temp.split(self.END, 1)
            if len(temp) > 1:
                text += temp[0]
                self.cache = temp[1]
                break
            else:
                text += temp[0]
            temp = self.sock.recv(4096).decode('utf8')
        return text

if __name__ == '__main__':
    host = '127.0.0.1'
    port = 8888
    client = EchoClient(host, port)
    text = 'hello world!\r\nhello all?'
    client.send(text)
    print('> send:', text, sep='\n')
    text = client.recv()
    print('> read:', text, sep='\n')
```

输出为:
```
> send:
hello world!
hello all?
> read:
hello world!\r\nhello all?\r\n\r\n
```

## 参考资料:
- [socket Low-level networking interface](https://docs.python.org/3/library/socket.html)
- [Interprocess Communication and Networking](https://docs.python.org/3/library/ipc.html)