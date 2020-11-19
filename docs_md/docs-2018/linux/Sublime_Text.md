# Sublime Text
Sublime Text是开发代码编辑的神器，编辑器界面优美，操作速度快速，还支持插件扩展机制，用她来写代码，绝对是一种享受。[Sublime Text 3](https://www.sublimetext.com/3)是Sublime Text的当前版本。

## 安装
Windows直接下载二进制安装文件，Ubuntu会稍麻烦一点。下面演示[Linux repos](https://www.sublimetext.com/docs/3/linux_repositories.html)安装：
```
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
```

有时你希望或被迫使用[?.tar.bz2](https://www.sublimetext.com/3)安装：
```
tar -jxf ?.tar.bz2 -C /opt
/opt/sublime_text_3/sublime_text
```

觉得命令运行麻烦，执行`cp /opt/sublime_text_3/sublime_text.desktop /usr/share/applications`。

## 偏好设置
[Package Control](https://packagecontrol.io/installation)官网选择对应的Sublime Text版本然后复制安装代码。按`Ctrl+~`调出控制台，粘贴安装代码到底部命令行并回车。可能需要重启Sublime text，如果在`Preferences->package settings`中看到`package control`这一项，则安装成功。

Sublime中将TAB缩进转化为4个空格。打开用户配置文件`Preferences->Settings`，添加以下内容：
```
{
    "translate_tabs_to_spaces": true,
    "default_line_ending": "unix",
    "show_line_endings": true,
    "show_encoding": true,
    "tab_size": 4
}
```

## 功能扩展
Sublime常用扩展推荐：

- MarkdownEditing，Markdown文件编辑
- SideBarEnhancements，边栏菜单功能扩充
- AllAutocomplete，自动补全会搜索全部打开的标签页，这将极大的简化开发进程
- Sublimerge 3，快捷键`Ctrl+Alt+D`，选择`Compare to View...`，选择文件即可对比文件
- Terminal，打开终端，`Ctrl+Shift+T`在文件所在目录，`Ctrl+Shift+Alt+T`在文件所在项目根目录
- Markdown Preview，`Ctrl+Shift+P`调出命令面板，输入`mdp`，选择`Markdown Preview: Preview in Browser`预览
- Pretty JSON，`Ctrl+Alt+J`美化整个JSON文件，`Ctrl+Alt+M`压缩JSON文件
- sublack，Python代码格式化程序[black](https://github.com/python/black)的集成，`Ctrl+Alt+B`格式化整个文件，`Ctrl+Alt+Shift+B`在新选项卡中显示差异

### Terminal
替换系统默认终端，修改配置`/Terminal/Settings – User`：
```
{
    // See https://github.com/wbond/sublime_terminal#examples
    "terminal": "C:\\Program Files\\Git\\git-bash.exe",
    "parameters": [],
    "env": {}
}
```

>在win10系统中终端启动失败，建议以管理员身份启动Sublime。

## 常用快捷键
- 鼠标选中多行，按下`Ctrl+Shift+L`可同时编辑这些行
- 鼠标选中文本，反复按`CTRL+D`可同时选中下一个相同的文本
- 鼠标选中文本，按下`Alt+F3`可一次性选择全部的相同文本
