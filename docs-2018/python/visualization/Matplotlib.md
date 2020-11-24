# Matplotlib

## backends
有时候我们需要matplotlib强大的绘图能力,但不希望直接在Jupyter中显示,而是输出到文件:
```python
# object-oriented plot
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

fig    = Figure()
canvas = FigureCanvas(fig)

# first axes
ax1    = fig.add_axes([0.1, 0.1, 0.2, 0.2])
line   = ax1.plot([0,1], [0,1])
ax1.set_title("ax1")

# second axes
ax2    = fig.add_axes([0.4, 0.3, 0.4, 0.5])
sca    = ax2.scatter([1,3,5], [2,1,2])
ax2.set_title("ax2")

canvas.print_png("temp.png")
```

## 中文测试
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
my_x = np.arange(50)
my_y1 = np.random.randn(50).cumsum()
my_y2 = np.random.randn(50).cumsum()

plt.plot(my_x, my_y1, label='类别1')
plt.plot(my_x, my_y2, label='类别2')
plt.title("中文标题")
plt.legend(loc='best')
plt.show()
```

### Windows
查看`matplotlibrc`配置文件位置:
```python
import matplotlib
print(matplotlib.matplotlib_fname())
#site-packages/matplotlib/mpl-data/matplotlibrc
```

依据显示的路径,修改`matplotlibrc`配置文件:
```
line:196 => font.family         : DengXian
line:208 => font.sans-serif     : Dengxian
```

### Ubuntu
首先清空缓存`~/.cache/matplotlib`,拷贝`C:\Windows\Fonts`目录`等线`字体文件`Deng.ttf`到`matplotlib/mpl-data/fonts/ttf/`,并修改`matplotlibrc`配置文件：
```
line:196 => font.family         : DengXian
line:208 => font.sans-serif     : Dengxian
```

查看`matplotlibrc`配置文件位置：
```python
import matplotlib
print(matplotlib.matplotlib_fname())
#site-packages/matplotlib/mpl-data/matplotlibrc
```

## init
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%pylab inline
```

可选的`style`：
```
plt.style.use('ggplot')
print(plt.style.available)
```

## Save Image
利用`plt.savefig`可以将当前图表保存到文件，文件类型是通过文件扩展名推断出来的。发布图片时最常用到的两个重要选项是`dpi`和`bbox_inches`（可以剪除当前图表周围的空白部分）。要得到一张带有最小最小白边且分辨率为400dpi的png图片：

```
plt.savefig('fig_path.png', dpi=400, bbox_inches='tight');
```

## 统计绘图
灵活绘制单变量观测分布：
```
import numpy as np
import seaborn as sns

np.random.seed(0)
x = np.random.randn(100)
sns.distplot(x, rug=True, hist=True)
```
