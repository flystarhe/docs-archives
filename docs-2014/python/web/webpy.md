title: 使用webpy提供web服务
date: 2017-08-07
tags: [Python]
---
webpy是快速搭建web服务的利器,不过现在不推荐使用了.官网不怎么更新了,也不支持py35.

<!--more-->
## 安装webpy
执行命令`pip install web.py`,或:
```
git clone git://github.com/webpy/webpy.git
```

## 搭建服务
使用webpy搭建web服务,`hej.py`:
```
import web

web.config.debug = False

urls = ('/index', 'index',
        '/guess', 'guess')

class index:
    def GET(self):
        return 'Class: index'

class guess:
    def POST():
        data = web.data()
        return str(data)

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()
```

从form或url参数接受用户数据:
```
class RequestHandler:
    def GET(self, name):
        return 'name: {0}'.format(name)

class RequestHandler:
    def GET(self):
        user_data = web.input()
        return 'name: %s' % user_data.name

class RequestHandler():
    def POST():
        data = web.data()
```

## 启动服务
命令行输入`python hej.py`,也可以指定端口号:
```
python hej.py 9090
```

## 测试服务
测试GET方法:
```
import requests

url = 'http://127.0.0.1:9090/guess'
vals = {'data': '[[1,"name","name_"],[2,"name","name_"]]'}

response = requests.get(url, params=vals)
print(response.status_code)
print(response.url)
print(response.text)
```

测试POST方法:
```
import requests

url = 'http://127.0.0.1:9090/guess'
vals = {'data': '[[1,"name","name_"],[2,"name","name_"]]'}

response = requests.post(url, data=vals)
print(response.status_code)
print(response.url)
print(response.text)
```

## 参考资料：
- [webpy.org](http://webpy.org/)
- [web.py 0.3 新手指南](http://webpy.org/docs/0.3/tutorial.zh-cn)
- [Web.py Cookbook 简体中文版](http://webpy.org/cookbook/index.zh-cn)