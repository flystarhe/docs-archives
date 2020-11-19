title: Hexo搭建自己的博客
date: 2014-12-14
tags: [Git,Hexo]
---
Hexo出自台湾大学生tommy351之手，是一个基于Node.js的静态博客程序，其编译上百篇文字只需要几秒。hexo生成的静态网页可以直接放到GitHub Pages，BAE，SAE等平台上。

<!--more-->
## 安装
[安装Hexo](https://hexo.io/zh-cn/docs/)相当简单。然而在安装前，您必须检查电脑中是否已安装下列应用程序：

- [Node.js](http://nodejs.org/)
- [Git](http://git-scm.com/)

输入`node -v`、`npm -v`，若有版本返回，表示安装成功。

### 安装 Node.js
安装`Node.js`的最佳方式是使用[nvm](https://github.com/creationix/nvm):
```
wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.33.2/install.sh | bash
nvm install stable
```

### 安装 Git

- Windows：下载并安装[git](https://git-scm.com/download/win)
- Mac：使用 Homebrew, MacPorts：`brew install git`
- Linux (Ubuntu, Debian)：`sudo apt-get install git-core`
- Linux (Fedora, Red Hat, CentOS)：`sudo yum install git-core`

### 安装 cnpm
安装淘宝的`npm`镜像加速，这大部分`npm`命令可以用`cnpm`替代：
```
npm install -g cnpm --registry=https://registry.npm.taobao.org
```

如果您的电脑中已经安装上述必备程序，只需要使用`npm`即可完成`Hexo`的安装，也可以使用`cnpm`：
```
npm install hexo-cli -g
```

## 建站
安装`Hexo`完成后，请执行下列命令，将会在指定文件夹中新建所需要的文件：
```
cd /home/hejian/work_git/
hexo init blog_2014 && cd blog_2014
npm install
hexo g
hexo s
```

新建完成后，指定文件夹的目录如下：
```
# .
# ├── _config.yml
# ├── package.json
# ├── scaffolds
# ├── source
# |   ├── _drafts
# |   |── _posts
# |── themes
```

### _config.yml
网站的`配置`信息，您可以在此配置大部分的参数。

- title，网站标题
- subtitle，网站副标题
- description，网站描述，主要用于SEO
- author，您的名字
- language，网站使用的语言
- timezone，网站时区
- url，网址
- root，网站根目录
- permalink，文章的永久链接格式`:year/:month/:day/:title/`
- permalink_defaults，永久链接中各部分的默认值

### source
资源文件夹是存放用户资源的地方。除`_posts`文件夹之外，开头命名为`_(下划线)`的文件、文件夹和隐藏的文件将会被忽略。`Markdown`和`HTML`文件会被解析并放到`public`文件夹，而其他文件会被拷贝过去。

### themes
`主题`文件夹。`Hexo`会根据主题来生成静态页面。

## 扩展
切换到博客根目录，安装扩展：
```
npm install hexo-math --save
npm install hexo-deployer-git --save
npm install hexo-generator-searchdb --save
```

## 主题
切换到博客根目录，安装主题：
```
git clone https://github.com/xiangming/landscape-plus.git themes/landscape-plus
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

- [站点·配置文件·参考](readme01.txt)
- [主题·配置文件·参考](readme02.txt)

## 笔记
`hexo g`生成静态页面，失败则删除`db.json`后重试。`hexo s`启动本地服务，浏览器输入[http://localhost:4000](http://localhost:4000)查看效果。

### LaTex
LaTex的保留字符有`# $ % & ~ _ ^ \ { }`。若要在数学环境中表示这些符号，需要分别表示为`\# \$ \% \& \~ \_ \^ \{ \}`，即在字符前加上`\`。反斜杠`\`比较特殊`\backslash`。

### Linux
```
cd /home/hejian/work_git/blog_2014/source/
rm -rf ./*
\cp -rf /home/hejian/work_main/blog_2014/* .

cd /home/hejian/work_git/blog_2014/
hexo clean
hexo g
hexo d
```

## 参考资料：
- [hexo themes list](https://github.com/hexojs/hexo/wiki/Themes)
- [Generating a new SSH key and adding it to the ssh-agent](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)
- [Adding a new SSH key to your GitHub account](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/)
- [latex数学公式](http://hustlei.tk/2014/08/latex-math-equation.html)