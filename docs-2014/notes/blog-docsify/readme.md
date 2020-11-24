title: Docsify
date: 2018-03-30
tags: [Git,Docsify]
---
Docsify是基于Markdown的快速生成文档网站工具.官方特性有:无需构建,写完Markdown直接发布;支持自定义主题;容易使用并且轻量.然而,不支持数学公式,并且导航来不能自动生成.不是很推荐,偏娱乐.

<!--more-->
## 开始
安装[docsify](https://docsify.js.org/#/?id=docsify):
```bash
sudo apt install npm
sudo apt install nodejs
sudo apt install nodejs-legacy
sudo npm config set registry https://registry.npm.taobao.org
sudo npm config get registry
sudo npm i docsify-cli -g
```

初始化网站:
```bash
cd /data2/tmps/tmps/pages_docsify
#Initialize
docsify init ./docs
```

查看`./docs`目录文件列表:

- `index.html`,入口
- `README.md`,主页
- `.nojekyll`,防止忽略下划线开头的文件

可以更新文档`./docs/README.md`,当然也可以添加[更多页面](#).预览网站:
```bash
docsify serve ./docs
```

浏览器打开: http://localhost:3000

如果您不喜欢`npm`或无法安装,则可以手动创建`index.html`:
```html
<!-- index.html -->

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Document</title>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="description" content="Description">
  <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
  <link rel="stylesheet" href="//unpkg.com/docsify/lib/themes/vue.css">
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      loadSidebar: true,
      subMaxLevel: 2
    }
  </script>
  <script src="//unpkg.com/docsify/lib/docsify.min.js"></script>
</body>
</html>
```

使用Python运行静态服务器来预览:
```bash
cd docs
python -m http.server 3000
```

浏览器打开: http://localhost:3000

## 更多页面
如果您需要更多页面,则可以创建更多`md`文件.如果您创建了一个名为`guide.md`的文件,那么它可以通过`/#/guide`访问.

例如,目录结构如下所示:
```
.
└── docs
    ├── README.md
    ├── guide.md
    └── zh-cn
        ├── README.md
        └── guide.md
```

那么,文件路由:
```
docs/README.md        => http://domain.com
docs/guide.md         => http://domain.com/guide
docs/zh-cn/README.md  => http://domain.com/zh-cn/
docs/zh-cn/guide.md   => http://domain.com/zh-cn/guide
```

## 侧边栏
首先,你需要设置`loadSidebar`为`true`:
```html
<!-- index.html -->

<script>
  window.$docsify = {
    loadSidebar: true
  }
</script>
<script src="//unpkg.com/docsify/lib/docsify.min.js"></script>
```

创建`_sidebar.md`:
```markdown
<!-- docs/_sidebar.md -->

* [Home](/)
* [Guide](guide.md)
```

`_sidebar.md`从目录加载.如果当前目录不存在`_sidebar.md`,它将回退到父目录.例如,如果当前路径是`/guide/quick-start`,`_sidebar.md`将从中加载`/guide/_sidebar.md`.您可以指定`alias`以避免不必要的回退:
```html
<!-- index.html -->

<script>
  window.$docsify = {
    loadSidebar: true,
    alias: {
      '/.*/_sidebar.md': '/_sidebar.md'
    }
  }
</script>
```

## 目录
创建`_sidebar.md`后,侧边栏内容会根据`md`文件的标题自动生成.自定义边栏也可以通过设置`subMaxLevel`:
```html
<!-- index.html -->

<script>
  window.$docsify = {
    loadSidebar: true,
    subMaxLevel: 2
  }
</script>
<script src="//unpkg.com/docsify/lib/docsify.min.js"></script>
```
