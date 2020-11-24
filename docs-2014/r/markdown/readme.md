title: RMarkdown使用
date: 2017-05-22
tags: [R,Markdown]
---
RMarkdown编织叙事文本和代码讲述你与数据的故事，产生优雅的格式化输出。RMarkdown支持静态和动态的输出格式，包括HTML、PDF、Word、投影仪、HTML5幻灯片和更多格式。这里使用RStudio Server，浏览器输入`http://localhost:8787`进入集成开发环境：

1. `File > New File > RMarkdown`
2. 输入文档标题、作者以及输出的文档格式(html/pdf/word)

注：输出html或word都比较顺利，但是pdf时遇到了些问题。`Tools > Global Options > Code`设置`Default text encoding`为`UTF-8`，`Tools > Global Options > Sweave`设置`Weave Rnw files using`为`knitr`，设置`Typeset LaTeX into PDF using`为`XeLaTeX`。

<!--more-->
## Codes

### Inline Code
```r
Two plus two equals `r 2+2` .
```

### Code Chunks
使用`{r}`标识，支持`include, message, warning, eval, echo`等参数。
```r
dim(iris)
```

## Render
```r
library(rmarkdown)
base::date()
rmarkdown::render("test.Rmd", encoding="utf-8", params=list(), output_format="github_document", output_file="test_out.html")
base::date()
```

## Pdf

### Windows
虽然这里使用的是`MiKTeX`，不过推荐[TeXLive](http://www.tug.org/texlive)，Aliyun镜像[texlive2016.iso](http://mirrors.aliyun.com/CTAN/systems/texlive/Images/)。

TeXLive套件和MiKTeX相比，安装的包可能多一些，MiKTeX更轻量一些。此外，MiKTeX只支持Windows系统，而TeXLive支持Windows、Linux以及MacOS。当遇到`! LaTeX Error: File *.sty not found.`时，则安装相应软件包，如：
```
tlmgr update --self
tlmgr install titling
```

安装[MiKTeX](https://miktex.org/download)，启动RStudio，新建`R Markdown`文件，选择`Document`类型，`Default Output Format`设置为`PDF`，点击`OK`完成。有了这个`test.Rmd`文件，我们就可以执行`Knit > Knit to PDF`生成PDF文件。

也许不会那么顺利，错误报告`! LaTeX Error: File mathspec.sty not found.`，提示需要`mathspec`包。打开`MiKTeX Package Manager`界面，安装`mathspec`包。

使用`MiKTeX Package Manager`默认的`Repository`会失败。点击`Repository`菜单，打开`Change Package Repository`对话框，选择`Packages shall be installed from the internet`，点击`下一步`，选择`Japan ftp.jaist.ac.jp`，点击`完成`。

再次安装`mathspec`包，成功！再次执行`Knit > Knit to PDF`，成功生成PDF文件！

别高兴太早，不信加几个汉字试试。接下来我们解决中文问题：

- 新建`tmpl/header.tex`，内容如下：

```
\usepackage{xeCJK}
\setCJKmainfont{宋体}
\setmainfont{Georgia}
\setromanfont{Georgia}
\setmonofont{Courier New}
```

- 修改`test.Rmd`文件头，内容如下：

```
---
title: "index"
output:
    pdf_document:
        includes:
            in_header: tmpl/header.tex
        latex_engine: xelatex
---
```

### Linux
安装[texlive2016.iso](http://mirrors.aliyun.com/CTAN/systems/texlive/Images/)：
```
[root@fly _lib]# rm -rf /usr/local/texlive
[root@fly _lib]# rm -rf ~/.texlive2016
[root@fly _lib]# wget http://mirrors.aliyun.com/CTAN/systems/texlive/Images/texlive2016.iso
[root@fly _lib]# yum -y install perl-Digest-MD5 perl-Tk
[root@fly _lib]# mount -o loop texlive2016.iso /mnt/
[root@fly _lib]# cd /mnt
[root@fly mnt]# ./install-tl
[root@fly mnt]# cd
[root@fly ~]# umount /mnt/
[root@fly ~]# echo $'export PATH=/usr/local/texlive/2016/bin/x86_64-linux:${PATH}' >> /etc/profile
[root@fly ~]# source /etc/profile
[root@fly ~]# tlmgr update --self --repository http://mirrors.aliyun.com/CTAN/systems/texlive/tlnet/
[root@fly ~]# tlmgr update --all --repository http://mirrors.aliyun.com/CTAN/systems/texlive/tlnet/
```

启动RStudio，新建`R Markdown`文件，选择`Document`类型，`Default Output Format`设置为`PDF`，点击`OK`完成。有了这个`test.Rmd`文件，我们就可以执行`Knit > Knit to PDF`生成PDF文件。

同样不是很顺利，错误报告`! LaTeX Error: File framed.sty not found.`，安装`framed`包：
```
[root@fly ~]# tlmgr update --self
[root@fly ~]# tlmgr install framed
[root@fly ~]# tlmgr install titling
```

或使用图形界面`tlmgr --gui --gui-lang zh_CN`，再次执行`Knit > Knit to PDF`，成功生成PDF文件！

### Mac
安装[MacTeX](http://www.tug.org/mactex/)，启动RStudio，新建`R Markdown`文件，选择`Document`类型，`Default Output Format`设置为`PDF`，点击`OK`完成。有了这个`test.Rmd`文件，我们就可以执行`Knit > Knit to PDF`生成PDF文件。

同样不是很顺利，错误报告`! LaTeX Error: File framed.sty not found.`，安装`framed`包：
```
flystarhedeMacBook-Air:~ flystarhe$ sudo tlmgr update --self
flystarhedeMacBook-Air:~ flystarhe$ sudo tlmgr install framed
flystarhedeMacBook-Air:~ flystarhe$ sudo tlmgr install titling
```

再次执行`Knit > Knit to PDF`，成功生成PDF文件！

## Tmpl

### pdf_document
```
---
title: "index"
output:
    rticles::ctex:
        fig_caption: yes
        number_sections: yes
        toc: yes
documentclass: ctexart
classoption: "hyperref,"
---
```

### html_document
```
---
title: "index"
output:
    html_document:
        toc: true
        toc_depth: 5
        number_sections: true
        theme: paper
        highlight: tango
        mathjax: "http://example.com/mathjax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
---
```

### markdown_document
```
---
title: "index"
output:
    md_document:
        toc: true
        toc_depth: 5
        variant: markdown_github
---
```

### github_document
```
---
title: "index"
output:
    github_document:
        toc: true
        toc_depth: 5
---
```

### ioslides_presentation
```
---
title: "index"
output: ioslides_presentation
---
```

### beamer_presentation
```
---
title: "index"
output: beamer_presentation
---
```

## rticles
`rticles`包为各种格式提供了一套定制的`R Markdown LaTeX`格式和模板，执行命令`install.packages("rticles", type="source")`即可安装，使用`New R Markdown`对话框从`From Template`选择模板创建文章。

比如：`CTeX Documents`模板，也输出pdf文件，还直接支持中文。`GitHub Document(Markdown)`模板，可以输出`.md`文件，资源文件放在`*_files`。

## html2pdf
输出pdf的道路崎岖满满的，输出html的道路就阳光多了。不妨迂回一下，输出html文件，再用chrome打印，`目标打印机`选择`另存为pdf`，`选项`勾选`背景图形`，输出的pdf文件也不赖。比如，我就经常在`http://marxi.co/`写`.md`，导出html文件，再用chrome输出pdf。

## Parameters
在文档顶部的YAML部分中使用`params`字段声明参数，例如：
```
---
title: "index"
output: github_document
params:
    data: 0
    b: "string"
---
```
声明的参数在环境中自动提供，使用以下R代码访问参数：
```r
params$data
str(params)
```
使用`rmarkdown::render`函数的传递参数，例如：
```r
rmarkdown::render("py_test.Rmd", encoding="utf-8", params=list(data="Hi!"), output_format="github_document", output_file="py_test_out.html")
```

## Options
`knitr`的设置项非常有用，比如`markdown_document`默认输出图片到`Rmd_file_name_files/figure-markdown_github/`，而我希望输出到当前目录并以`Rmd_file_name.md.`为前缀：
```r
knitr::opts_chunk$set(fig.path=gsub("[rR]md$", "md.", knitr::current_input()))
```

更多内容可执行`knitr::opts_current$get()`来查看。

## 参考资料：
- [Introduction](http://rmarkdown.rstudio.com/lesson-1.html)
- [HTML Documents](http://rmarkdown.rstudio.com/html_document_format.html)
- [PDF Documents](http://rmarkdown.rstudio.com/pdf_document_format.html)
- [HTML5 slides](http://rmarkdown.rstudio.com/ioslides_presentation_format.html)
- [MathJax](https://www.mathjax.org/)