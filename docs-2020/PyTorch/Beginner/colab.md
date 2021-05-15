# Colab
在此示例中，我们将演示如何在Google Colab中运行代码。

## 本地文件系统
从本地文件系统上传文件。`files.upload`会返回已上传文件的字典。此字典的键为文件名，值为已上传的数据。
```
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
```

将文件下载到本地文件系统。`files.download`会通过浏览器将文件下载到本地计算机。
```
from google.colab import files

with open('example.txt', 'w') as f:
    f.write('some content')

files.download('example.txt')
```

## Google云端硬盘
以下示例展示了如何使用授权代码在您的运行时上装载Google云端硬盘，以及如何在那里写入和读取文件。一旦执行，您便可以在[drive.google.com](https://drive.google.com/)看到相应的新文件`foo.txt`。（请注意，此操作仅支持读取、写入和移动文件。）
```
from google.colab import drive
drive.mount('/content/gdrive')

with open('/content/gdrive/My Drive/foo.txt', 'w') as f:
    f.write('Hello Google Drive!')
!cat /content/gdrive/My\ Drive/foo.txt

drive.flush_and_unmount()
```

我们会将所需文件复制到我们的Google云端硬盘帐户中。

- 登录到Google云端硬盘。
- 创建`data/colab_0314_1014`文件夹。
- 下载对话语料库[cornell_movie_dialogs_corpus.zip](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)。
- 解压缩到本地计算机上，将`movie_lines.txt`复制到Google云端硬盘新建文件夹中。

现在，在Colab中指向上传到云端硬盘的文件。
```
import os
from google.colab import drive
drive.mount('/content/gdrive')

corpus = os.path.join("/content/gdrive/My Drive/data", "colab_0314_1014")

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus, "movie_lines.txt"))
drive.flush_and_unmount()
```

现在，当您单击“代码”部分的“运行”单元格按钮时，系统将提示您授权Google云端硬盘，并获得授权码。将代码粘贴到Colab中的提示中，就可以设置好了。

## Importing Libraries
要导入默认情况下不在Colaboratory中的库，可以使用`!pip install -q`或`!apt-get -qq install -y`。

7zip reader:
```
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -q -U libarchive
import libarchive
```

GraphViz & PyDot:
```
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install -q pydot
import pydot
```

TensorFlow 2 in Colab:
```
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
```

Tensorflow with GPU:
```
%tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print(
        '\n\nThis error most likely means that this notebook is not '
        'configured to use a GPU.  Change this in Notebook Settings via the '
        'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
    raise SystemError('GPU device not found')

def cpu():
    with tf.device('/cpu:0'):
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)

def gpu():
    with tf.device('/device:GPU:0'):
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)

# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
```

## 参考资料：
- [colab.research.google.com/notebooks](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
- [Browsing GitHub Repositories from Colab](https://colab.research.google.com/github/)
- [External data: Local Files, Drive, Sheets, and Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb?hl=en)
- [TPUs in Colab](https://colab.research.google.com/notebooks/tpu.ipynb)