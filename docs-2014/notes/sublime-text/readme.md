title: Sublime Text入门
date: 2015-05-20
tags: [Sublime]
---
Sublime text是开发代码编辑的神器，编辑器界面优美，操作速度快速。Sublime Text不仅具有华丽的界面，还支持插件扩展机制，用她来写代码，绝对是一种享受。

<!--more-->
## 安装Package Control
按`Ctrl + ~`调出控制台，粘贴安装代码到底部命令行并回车。重启Sublime text，如果在`Preferences->package settings`中看到`package control`这一项，则安装成功。

>Package Control[官网](https://packagecontrol.io/installation)选择对应的Sublime Text版本然后复制安装代码。

## Sublime中将TAB缩进转化为4个空格
打开用户配置文件`Preferences -> Settings – User`，添加以下内容：
```
{
    "font_face": "DejaVu Sans Mono",
    "font_size": 12,
    "translate_tabs_to_spaces": true,
    "tab_size": 4
}
```

## Sublime常用扩展推荐
- ConvertToUTF8，中文支持。
- SideBarEnhancements，边栏菜单功能扩充。
- SublimeTmpl，常用语言文件模板。
- MarkdownEditing，Markdown文件编辑。
- DocBlockr，输入`/**`按下Tab键会自动解析函数并且为你准备好合适的模板。
- AllAutocomplete，自动补全会搜索全部打开的标签页，这将极大的简化开发进程。
- GitGutter，可以高亮相对于上次提交有所变动的行，换句话说是实时的diff工具。
- SublimeREPL，可以直接在编辑器中运行一个解释器，支持很多语言。
- PlainTasks，杰出的待办事项表！可以创建项目，贴标签，设置日期。
- IPythonNotebook，暂无描述。

### 文件对比：Sublimerge 3
快捷键`Ctrl+Alt+D`，选择`Compare to View..`，选择文件即可对比文件。

### ChineseLocalizations
请使用主菜单的`帮助/Language`子菜单来切换语言。目前支持简体中文|繁体中文|日本語。要换回英语不需要卸载本插件，请直接从菜单切换英文。

### Terminal
`ctrl+shift+t`打开终端,在文件所在文件夹.`ctrl+shift+alt+t`打开终端,在文件所在项目根目录文件夹.

### Markdown Preview
Markdown Preview不仅支持在浏览器中预览markdown文件，还可以导出html代码。组合键`Ctrl+Shift+P`或`Preference->Package Control`调出命令面板，输入`mdp`，选择`Markdown Preview: Preview in Browser`在浏览器中预览`markdown`文件。

选中后，你将见到两个选项：`GitHub`和`Markdown`。GitHub选项意味着使用GitHub的在线API来解析`.md`文件，它的解析速度取决于你的联网速度。

另外一个常用功能`Export HTML in Sublime Text`，即导出html文件，等同快捷键`CTRL + B`。在最前面添加`[TOC]`自动生成目录，不过仅`Markdown`选项支持。

Sublime Text支持自定义快捷键，Markdown Preview默认没有快捷键，我们可以自己为`Preview in Browser`设置快捷键。方法是在`Preferences->Key Bindings`打开的`User`文件的中括号中添加以下代码：
```
{ "keys": ["ctrl+p"], "command": "markdown_preview", "args": {"target": "browser", "parser": "markdown"} }
```

这里`ctrl+p`可设置为自己喜欢的按键，`parser`也可设置为`github`，改为使用Github在线API解析`.md`。

设置`mathjax`支持需要在`Preferences->Package Settings->Markdown Preview->Settings - User`中增加如下代码：
```
{
    "enable_mathjax": true,
    "enable_highlight": true
}
```

## 搭建开发环境
对于Sublime原生不支持的开发语言需要我们动动手，点击菜单`Tools -> Build System -> New Build System...`打开空`sublime-build`配置文件，填写配置信息并保存为`语言.sublime-build`即可。下面是R、PHP和Scala语言的配置文件内容：

### R.sublime-build
```
    {
        "cmd": ["D:/Program Files/R/R-3.1.1/bin/x64/Rscript", "$file"],
        "selector": "source.r, source.R",
        "encoding": "utf-8"
    }
```

### PHP.sublime-build
```
    {
        "cmd": ["D:/php5/php", "$file"],
        "selector": "source.php",
        "encoding": "utf-8"
    }
```

### Scala.sublime-build
```
    {
        "cmd": ["C:/Program Files (x86)/scala/bin/scala.bat", "$file"],
        "selector": "source.scala",
        "encoding": "utf-8"
    }
```

## 强大的多行选择和多行编辑
- 鼠标选中多行，按下`Ctrl+Shift+L`可同时编辑这些行；
- 鼠标选中文本，反复按`CTRL+D`可同时选中下一个相同的文本进行同时编辑；
- 鼠标选中文本，按下`Alt+F3`可一次性选择全部的相同文本进行同时编辑；
- `Shift+鼠标右键`(或使用鼠标中键)可以用鼠标进行竖向多行选择；
- `Ctrl+鼠标左键`可以手动选择同时要编辑的多处文本。

## 参考资料：
- [Sublime Text非官方文档](http://sublime-text.readthedocs.org/en/latest/index.html)
- [Sublime Text 2 入门及技巧](http://lucifr.com/2011/08/31/sublime-text-2-tricks-and-tips/)