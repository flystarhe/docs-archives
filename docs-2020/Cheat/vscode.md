# VS Code

```
{
    "terminal.integrated.shell.windows": "C:\\Program Files\\Git\\bin\\bash.exe",
    "editor.wordWrap": "on",
    "python.formatting.autopep8Args": [
        "--max-line-length",
        "80"
    ],
    "editor.formatOnSave": true,
    "python.linting.pylintArgs": [
        "--generated-members=numpy.*,torch.*,cv2.*"
    ],
    "files.eol": "\n",
    "editor.fontSize": 18
}
```

* `F1` - 命令面板
* `F12` - 转到定义
* `Alt+F12/Option+F12` - 窥视定义
* `Shift+F12` - 查看光标所在函数或变量的引用
* `Alt+左/右箭头` - 前进或者后退到光标所在源码的上一个位置
* `Control+-/Control+Shift+-` - 前进或者后退到光标所在源码的上一个位置
* `Ctrl+P/Command+P` - 快速打开文件列表，输入关键字匹配文件，方便的在指定文件之间跳转

## Refactoring

* 提取变量。提取当前范围内所选文本的所有类似出现，并将其替换为变量。选择文本，右键选择`Extract Variable`。
* 提取方法。提取当前范围内所选表达式或块的所有类似出现，并将其替换为方法调用。选择表达式或块，右键选择`Extract Method`。
* 排序导入。排序`import`使用`isort`包将来自同一模块的特定导入合并到单个`import`语句中。无需选择，右键选择`Sort Imports`。

## Extensions

* Python
* Jupyter
* Bookmarks
* Todo Tree
* GitLens
* Git Graph
* Remote - SSH
* Remote - WSL
