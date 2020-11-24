title: 视频质量评价
date: 2017-10-25
tags: [Image]
---
了解一点质量评价方法.MSE,SNR,PSNR,SSIM,MOS.

<!--more-->
## 全参考视频客观质量评价
MSE,SNR,PSNR,SSIM.

### MSE
均方误差:

\begin{align}
MSE = \frac{\sum_{x=1}^{m} \sum_{y=1}^{n} (f(x,y)-\hat{f}(x,y))^2}{m*n}
\end{align}

- $f(x,y)$,表示原始信号/图像
- $\hat{f}(x,y)$,表示处理后的信号/图像

### SNR
信噪比:

\begin{align}
10*log_{10} \frac{\sum_{x=1}^{m} \sum_{y=1}^{n} f(x,y)^2}{\sum_{x=1}^{m} \sum_{y=1}^{n} (f(x,y)-\hat{f}(x,y))^2}
\end{align}

### PSNR
峰值信噪比:

\begin{align}
10*log_{10} \frac{\sum_{x=1}^{m} \sum_{y=1}^{n} 255^2}{\sum_{x=1}^{m} \sum_{y=1}^{n} (f(x,y)-\hat{f}(x,y))^2}
\end{align}

相比`SNR`,只是信号部分的值通通改用該信号度量的最大值.当以信号度量范围为`0~255`计算时,信号部分均当成是其能够度量的最大值,也就是255,而不是原來的信号.

用得最多,但是其值不能很好地反映人眼主观感受.一般取值范围`20~40`.值越大,视频质量越好.

### SSIM
SSIM(structural similarity index),结构相似性,是一种衡量两幅图像相似度的指标.计算稍复杂,其值可以较好地反映人眼主观感受.一般取值范围`0~1`.值越大,视频质量越好.它分别从亮度,对比度,结构三方面度量图像相似性:

\begin{align}
& l(X,Y) = \frac{2 \mu_X \mu_Y + C_1}{\mu_X^2 + \mu_Y^2 + C_1} \\
& c(X,Y) = \frac{2 \sigma_X \sigma_Y + C_2}{\sigma_X^2 + \sigma_Y^2 + C_2} \\
& s(X,Y) = \frac{\sigma_{XY} + C_3}{\sigma_X \sigma_Y + C_3}
\end{align}

其中$\mu_X , \mu_Y$分别表示图像X和Y的均值,$\sigma_X , \sigma_Y$分别表示图像X和Y的方差,$\sigma_X \sigma_Y$表示图像X和Y的协方差.

$C_1 , C_2 , C_3$为常数,为了避免分母为0的情况,通常取:

\begin{align}
& C_1 = (K1 * L)^2 \\
& C_2 = (K2 * L)^2 \\
& C_3 = C_2 / 2
\end{align}

一般地`K1=0.01`,`K2=0.03`,`L=255`.则:

\begin{align}
SSIM(X,Y) = l(X,Y) * c(X,Y) * s(X,Y)
\end{align}

在实际应用中,可以利用滑动窗将图像分块,令分块总数为N,考虑到窗口形状对分块的影响,采用高斯加权计算每一窗口的均值,方差以及协方差.然后计算对应块的结构相似度SSIM,最后将平均值作为两图像的结构相似性度量,即平均结构相似性[MSSIM](#):

\begin{align}
MSSIM(X,Y) = \frac{1}{N} \sum_{k=1}^{N} SSIM(X_k, Y_k)
\end{align}

### MOS
MOS(Mean Opnion Score,平均意见分)是主观评价实验之后,得到的主观分数,取值`0~100`.值越大,代表主观感受越好.
