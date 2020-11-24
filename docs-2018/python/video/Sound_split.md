# Sound split
分割录音的问题，将录音文件转化为小小的分段，主要是按照语音停顿进行分割。运用pydub插件对音频进行分割，分割方式是运用静音方式进行。

## Install
```bash
apt-get install ffmpeg
pip install pydub
```

## 静音分割
示例代码：
```python
from pydub.silence import split_on_silence

# 实现拆分，只要这一句代码就够了
chunks = split_on_silence(chunk, min_silence_len=700, silence_thresh=-70)
# silence_thresh是认定小于-70dBFS以下的为silence
# 发现小于-70dBFS部分超过700毫秒，就进行拆分
```

- `dBFS`是指数字音响中以满刻度为基准的分贝值，低于满刻度`20dB`的信号便是`-20dBFS`。

## 音频剪切
我们常常有一段音频的剪切这种需求，希望把他剪短只取一部分来用：
```python
from pydub import AudioSegment

file_name = "lesson01.mp3"
sound = AudioSegment.from_mp3(file_name)
ten_seconds = 10 * 1000
word = sound[:ten_seconds]
```

>剪切时间是按`ms`毫秒来的，所以时间格式的转换就要到毫秒级的。

## 声音分贝的概念

### dB(dBSPL)
声音本质上来说是一种波，通过空气传播，传到人耳朵里引发鼓膜的振动。所以，声音的大小，实际就是对这种振动强度的反映。而由于空气的振动会引起大气压强的变换，可以使用压强变化的程度来描述声音的大小，这就是“声压(SPL,Sound Pressure Levels)”概念，其单位是Pa。例如：1米外步枪射击的声音大约是7000Pa；10米外开过汽车大约是0.2Pa。

使用声压作为测量量的分贝就是dBSPL，通常用来表示声音大小的dB多说指的就是dBSPL。声压和声音大小的关系，可以使用如下公式表示：

$$
\begin{aligned}
I = \frac{P^2}{\rho}
\end{aligned}
$$

其中，I是声音的强度；P是声压；`\rho`是空气阻力，通常在室温下，空气阻力大约是400。

分贝的计算还需要选择一个特定的声压值作为“标准值”（0分贝），该值是固定的。有了这个基准值后代入上面的公式：

$$
\begin{aligned}
I(dB) = 10 \times \log_{10} \frac{P^2}{P_{ref}^2} = 20 \times \log_{10} \frac{P}{P_{ref}}
\end{aligned}
$$

其中，P是声压测量值；`P_{ref}`是标准值（0dBSPL）。这里选择的声压标准值为`2×10−5Pa`，是人耳在1KHz这个频率下能听到的最小的声音，大致相当于3米外一只蚊子在飞的声音。

### dBFS
在数字时代更多的音频分贝表示是dBFS。dBFS的全称为Decibels Full Scale，全分贝刻度，是数值音频分贝值的表示方法。dBFS的基准并不是最小的或者是中间的某一个值，是最大的那个值！也就是说0dBFS是数字设备能达到的最大值，除了最大值外都是负值。

以数字音频的sample为16位无符号为例，16位的无符号的最大值为65536，因此dBFS的计算公式：

$$
\begin{aligned}
dBFS = 20 \times \log_{10} \frac{sample}{65536}
\end{aligned}
$$

这样，最小的`-96dBFS|sample=1`。也就是说16位无符号音频的动态范围为`-96~0`。

## 参考资料：
- [pydub](https://github.com/jiaaro/pydub)
- [python音频处理库：pydub](http://appleu0.sinaapp.com/?p=588)