# Display server visdom
Visdom旨在促进(远程)数据的可视化,重点是支持科学实验.启动命令为:`python -m visdom.server`.

## 命令行选项
可以向服务器提供以下选项:

- `-port`:运行服务器的端口
- `-env_path`:加载序列化会话的路径
- `-readonly`:标记以只读模式启动服务器
- `-enable_login`:标记服务器为需要身份验证

## 用法
```python
import visdom
import numpy as np
viz = visdom.Visdom()
```

text demo:
```python
textwindow = viz.text("Hello World!")

updatetextwindow = viz.text("Hello World! More text should be here")
assert updatetextwindow is not None, "Window was none"
viz.text("And here it is", win=updatetextwindow, append=True)

# text window with Callbacks
txt = "This is a write demo notepad. Type below. Delete clears text:<br>"
callback_text_window = viz.text(txt)
def type_callback(event):
    if event["event_type"] == "KeyPress":
        curr_txt = event["pane_data"]["content"]
        if event["key"] == "Enter":
            curr_txt += "<br>"
        elif event["key"] == "Backspace":
            curr_txt = curr_txt[:-1]
        elif event["key"] == "Delete":
            curr_txt = txt
        elif len(event["key"]) == 1:
            curr_txt += event["key"]
        viz.text(curr_txt, win=callback_text_window)
viz.register_event_handler(type_callback, callback_text_window)
```

matplotlib demo:
```python
try:
    import matplotlib.pyplot as plt
    plt.plot([1, 23, 2, 4])
    plt.ylabel("some numbers")
    viz.matplot(plt)
except BaseException as err:
    print("Skipped matplotlib example")
    print("Error message: ", err)
```

image demo:
```python
viz.image(
    np.random.rand(3, 512, 256),
    opts=dict(title="Random!", caption="How random."),
)

# grid of images
viz.images(
    np.random.randn(20, 3, 64, 64),
    opts=dict(title="Random images", caption="How random.")
)
```

line plots:
```python
viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

Y = np.linspace(-5, 5, 100)
viz.line(
    Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
    X=np.column_stack((Y, Y)),
    opts=dict(markers=False),
)

# line updates
win = viz.line(
    X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
)
viz.line(
    X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
    Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
    win=win,
    update="append"
)
viz.line(
    X=np.arange(21, 30),
    Y=np.arange(1, 10),
    win=win,
    name="2",
    update="append"
)
```

## 参考资料:
- [example/demo.py](https://github.com/facebookresearch/visdom/blob/master/example/demo.py)