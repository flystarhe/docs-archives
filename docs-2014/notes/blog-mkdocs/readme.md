title: MkDocs
date: 2018-03-30
tags: [Git,MkDocs]
---
MkDocs是基于Markdown的快速生成文档网站工具.聪明的导航,简洁的外观,我很是喜欢.数学公式和中文搜索的支持更是完美.

<!--more-->
## 安装
[MkDocs](http://www.mkdocs.org/)安装很简单:
```bash
pip install --upgrade pip
pip install mkdocs
mkdocs -V
#mkdocs, version 0.17.3
pip install mkdocs-material
pip install pygments
pip install pymdown-extensions
```

## 入门
```bash
mkdocs new docs
cd docs
```

浏览创建的最初项目:
```
#tree -L 2 .
.
├── docs
│   └── index.md
└── mkdocs.yml
```

一个名为`mkdocs.yml`的配置文件,一个名为`docs`的文件夹,目前只包含一个名为`index.md`文档.

MkDocs带有一个内置的服务器,可让你在处理文档时预览文档.确保你与`mkdocs.yml`在同一目录,运行`mkdocs serve`.然后,浏览器打开[http://127.0.0.1:8000](http://127.0.0.1:8000).

>支持自动重新加载,只要配置文件,文档目录或主题目录中的任何内容发生更改,就会重新生成文档.

## 配置
现在网站标题还是`My Docs`,因此你需要编辑配置文件:
```
site_name: Hej
```

添加`pages`设置,可控制导航标题的顺序,将失去一些强大功能:
```
site_name: Hej
pages:
    - Home: index.md
    - Guid: readme.md
```

更改主题来更改文档的显示方式:
```
site_name: Hej
pages:
    - Home: index.md
    - Guid: readme.md
theme:
    name: readthedocs
```

保存更改,将看到正在使用的`readthedocs`主题,默认`mkdocs`.

这是我的配置,可供参考:
```
site_name: 'Docs 2014'
site_url: 'https://flystarhe.github.io/docs-2014'

site_dir: docs
docs_dir: docs_md

repo_name: 'flystarhe/docs-2014'
repo_url: 'https://github.com/flystarhe/docs-2014'

theme:
    name: 'material'
    language: 'zh'

markdown_extensions:
    - codehilite:
        linenums: true
        guess_lang: false
    - toc:
        permalink: true
    - pymdownx.arithmatex
    - pymdownx.betterem:
        smart_enable: all
    - pymdownx.caret
    - pymdownx.critic
    - pymdownx.details
    - pymdownx.emoji:
        emoji_generator: !!python/name:pymdownx.emoji.to_svg
    - pymdownx.inlinehilite
    - pymdownx.magiclink
    - pymdownx.mark
    - pymdownx.smartsymbols
    - pymdownx.superfences
    - pymdownx.tasklist:
        custom_checkbox: true
    - pymdownx.tilde

extra:
  search:
    language: 'en, jp'
    tokenizer: '[\s\-\.]+'

extra_javascript:
    - 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-MML-AM_CHTML'
```

## 编译
看起来不错,您已准备好部署.首先建立文档:
```
mkdocs build --clean
```

这将创建一个名为`site`的新目录.一段时间后,文件可能会从文档中删除,但它们仍将驻留在`site`目录中.要删除这些陈旧的文件,只需`mkdocs build --clean`.如果你参考了我的配置,你的输出目录将会是`docs`,`docs_md`才是你的`source`目录.

## 部署
在部署前需要先建仓库:
```
git init
git remote add origin git@github.com:flystarhe/docs-2014.git
mkdocs build --clean
git add .
git commit -m "init"
git push origin master
```

>在正式部署前请执行`ssh -T git@github.com`,验证`Connecting with SSH`是可以的,否则请先完成相关设置.

如果您在GitHub上托管项目的源代码,则可以轻松使用GitHub页面托管项目的文档.在您维护项目源文档的git存储库checkout的主要工作分支(通常master)之后,运行以下命令:
```
mkdocs gh-deploy --clean
```

使用`mkdocs gh-deploy --help`获得`gh-deploy`命令的可用选项的完整列表.

>如果还嫌麻烦,也可以设置`GitHub Pages`为`master branch /docs folder`,代替`mkdocs gh-deploy --clean`.

[Read the Docs](https://readthedocs.org/)提供免费的文档托管.您可以使用任何主要版本控制系统导入文档,包括Mercurial,Git,Subversion和Bazaar.按照其网站上的说明妥善安排存储库中的文件,创建一个帐户并将其指向您公开托管的存储库.如果配置正确,每次将提交提交到公共存储库时都会更新文档.