title: Python网络编程之上传下载
date: 2017-11-16
tags: [Python]
---
本教程演示,首先使用`Tornado`搭文件服务,然后使用`requests`上传文件.

<!--more-->
## Tornado接收文件
上传图片使用了表单提交,`enctype="multipart/form-data"`表示不对字节进行编码,`<input type="file" ..`指定上传类型:
```html
<form action="/" enctype="multipart/form-data" method="post">
  <input type="file" name="images">
</form>
```

Tornado接收文件:
```python
#coding=utf-8
import tornado.ioloop
import tornado.web

class Index(tornado.web.RequestHandler):
    def post(self):
        imgfile = self.request.files.get('images')
        for img in imgfile:
            # keys: filename, body, content_type
            tmpname = './tmp-{}'.format(img['filename'])
            with open(tmpname, 'wb') as f:
                f.write(img['body'])
        self.write(tmpname + '/' + img['content_type'])

from io import StringIO, BytesIO
from PIL import Image as pil_image
class Resize(tornado.web.RequestHandler):
    def post(self):
        size = int(self.request.headers.get('Content-Length'))
        if size > 2*1024*1024:
            self.write('上传图片不能大于2M.')
        else:
            imgfile = self.request.files.get('images')
            for img in imgfile:
                im = pil_image.open(BytesIO(img['body']))
                im = im.resize((72, 72))
                # like a file
                im_file = BytesIO()
                im.save(im_file, format='png')
                # image data
                im_data = im_file.getvalue()
                tmpname = './tmp-resize-{}'.format(img['filename'])
                with open(tmpname, 'wb') as f:
                    f.write(im_data)
            self.write(tmpname)

router = [(r'/test/upload', Index),
          (r'/test/resize', Resize)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

## requests上传文件
```python
import requests

url = 'http://127.0.0.1:9000/test/upload'
files = [('images', ('pic1.jpg', open('1.jpg', 'rb'), 'image/jpeg')),
         ('images', ('pic2.jpg', open('1.jpg', 'rb'), 'image/jpeg'))]

response = requests.post(url, files=files)
print(response.status_code)
print(response.url)
print(response.text)
```

## 图片服务小结
```python
import tornado.ioloop
import tornado.web
from io import BytesIO
from PIL import Image

class Index(tornado.web.RequestHandler):
    def get(self):
        filename = self.get_argument('image')
        # image
        img = Image.open('/data2/tmps/{}'.format(filename))
        imgformat = img.format
        # some work
        img = img.resize((100, 100))
        # tmpfile, like a file
        tmpfile = BytesIO()
        img.save(tmpfile, format=imgformat)
        # image data
        imgdata = tmpfile.getvalue()
        # write
        self.set_header('Content-type', 'image/{}'.format(imgformat.lower()))
        self.write(imgdata)

    def post(self):
        imglist = self.request.files.get('images')
        if imglist is None:
            self.write('no image')
            self.finish()
        imglist = imglist[0]
        imgname = imglist['filename']
        imgbody = imglist['body']
        imgtype = imglist['content_type']
        print('=>', imgname, type(imgbody), imgtype)
        # image
        img = Image.open(BytesIO(imgbody))
        imgformat = img.format
        # some work
        img = img.resize((100, 100))
        # tmpfile, like a file
        tmpfile = BytesIO()
        img.save(tmpfile, format=imgformat)
        # image data
        imgdata = tmpfile.getvalue()
        # write
        self.set_header('Content-type', 'image/{}'.format(imgformat.lower()))
        self.write(imgdata)

router = [(r'/', Index),
          (r'/image', Index)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
```

测试:
```python
import requests

url = 'http://127.0.0.1:9000/image'
files = [('images', ('pic1', open('1.png', 'rb'), 'image/png')),
         ('images', ('pic2', open('2.jpg', 'rb'), 'image/jpeg'))]

response = requests.post(url, files=files)
print(response.status_code)
print(response.headers['content-type'])
print(response.url)

from io import BytesIO
from PIL import Image
Image.open(BytesIO(response.content))
```

或者打开浏览器,请求[http://127.0.0.1:9000/image?image=1.png](http://127.0.0.1:9000/image?image=1.png).(注意,请确保`server.py`所在目录存在图片`1.png`)

## 参考资料:
- [Requests: HTTP for Humans](http://www.python-requests.org/en/master/)
- [Requests: 让 HTTP 服务人类](http://docs.python-requests.org/zh_CN/latest/)