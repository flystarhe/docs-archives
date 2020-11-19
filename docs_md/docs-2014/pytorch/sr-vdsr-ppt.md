title: 翻译 | VDSR PPT
date: 2017-10-25
tags: [SR]
---
翻译[VDSR PPT](http://cv.snu.ac.kr/research/VDSR/).该论文利用残差学习,加深网络结构(20层),在图像的超分辨率上取得了较好的效果.输入图片的大小不一致,使得网络可以针对不同倍数的超分辨率操作.还有一个小技巧,就是卷积后补0使得图像大小保持一致.

<!--more-->
## 创新点
1. 加深网络结构(20层):更深的网络结构使得后面的网络层拥有更大的感受野,该文章采取`3x3`的卷积核,深度为D的网络拥有`(2D+1)x(2D+1)`的感受野,从而可以根据更多的像素点去推断结果像素点
2. 残差学习:加深的网络结构为梯度传输带来了困难,采用残差学习,提高学习率,加快了收敛速度,同时采用调整梯度裁剪(Adjustable Gradient Clipping)
3. 数据混合:将不同大小倍数的图像混合在一起训练,从而支持不同倍数的高清化

## 卷积补0
首先,卷积有个好处,就是提高较大的感受野,感受野就是视野的意思.图片预测,如果利用一个像素点去推断一个像素点,那么是做不好的.所以就要用卷积,卷积使得可以根据`NxN`个像素去推断目标像素值,图像处理中大家普遍认为像素之间是有相关性的.所以,根据更多的像素数据去推断目标像素,也是认为是一个提高效果的操作.

但是图片通过逐步卷积,图像大小也越来越小.一个很小的图:首先,不适合一个很深的网络(这会要求输入图片的size很大);其次,如果采取传统的卷积操作,边界像素会因考虑的区域少,感受野小而导致结果较差.于是传统的做法是对卷积结果做一个裁剪,剪去边界区域,随之产生一个新的问题,裁剪后的图像变的更小.于是,作者提出一个新的策略,就是每次卷积后,图像的size变小.但是,在下一次卷积前,对图像进行补0操作,恢复到原来大小.这样不仅解决了网络深度的问题,同时对边界像素的预测结果也得到了提升.

## 问题陈述

### 单图像超分辨率 (SISR)

Enlarge an image with details recovered

放大图像,恢复细节

Estimate high-frequency information (i.e. edges, texture) that are lost

估计丢失的高频信息(即边缘,纹理)

Severely ill-posed inverse problem (many possible solutions)

严重的逆向问题(许多可能的解决方案)

Algorithms should exploit contextual information

算法应该利用上下文信息

Very challenging

非常有挑战性

## 常规方法

Self Similarity based method(SelfEx)

基于自相似的方法(SelfEx)

Dictionary-learning based method(A+)

基于词典学习的方法(A+)

Deep learning based method(SRCNN)

基于深度学习的方法(SRCNN)

## 我们的方法

### 动机

We found limitations in three aspects from existing SR methods (especially, from SRCNN)

我们从现有的SR方法(特别是SRCNN)发现了三个方面的局限性

Relies on the context of small image regions. SRCNN has only 3 layers (receptive field of 13x13), other methods use even smaller regions

依靠小图像区域的上下文.SRCNN只有3层(13x13的接受场),其他方法使用更小的区域

Training converges too slowly. SRCNN uses learning rate of 10-5. It takes several days to converge

训练收敛太慢.SRCNN使用10-5的学习率.收敛需要几天的时间

Works for only  a single scale. Most existing methods handle different upscaling problem independently

仅适用于单一尺度.大多数现有方法独立处理不同的升级问题

### 上下文的方法

Utilize contextual information spread over very large image regions. For a large scale factor, we need to see more

利用分布在非常大的图像区域的上下文信息.对于大规模的因素,我们需要看更多

We employ very deep CNN model

我们采用非常深的CNN模式

To keep pixel-wise info., avoid pooling

要保持像素方面的信息,避免汇集

### 快速收敛的方法

Residual image learning

残差学习

Gradient clipping

渐变剪辑

### 多尺度的方法

Train a single convolutional network to learn and handle multi-scale SR

训练一个卷积网络来学习和处理多尺度SR

Different scale helps each other!

不同规模有助于对方!

Learns to upscale with inter-scale factor!

了解高分辨率因子!

### 建议的模型

Exploits very deep CNN to SR

利用较深的CNN做SR

20 convolutional layers (41x41 receptive field)

20个卷积层(41x41接收场)

64 channels, 3x3 filters in each convolutional layer

64个通道,每个卷积层中有3x3个滤镜

Skip connection to learn residual only

跳过连接仅学习残差

No dimension reduction such as pooling

没有尺寸减少,如汇集

### 训练细节
Residual learning

残差学习

Given a training dataset

给定训练数据集

Define residual image. most values are likely to be zero or small

定义残差图像.大多数值可能是零或小

Goal is to minimize. r is target residual image, f is the network prediction, x is interpolated low-resolution image (i.e. bicubic), Final super-resolution result becomes  f(x) + x

目标是最小化.r是目标残差图像,f是网络预测,x是内插低分辨率图像(即双三次图像),最终超分辨率结果变为`f(x) + x`

SGD with momentum (0.9)

SGD动量0.9

Adjustable Gradient Clipping

可调梯度剪切

Adjustable gradient clipping is an efficient way to handle exploding / vanishing gradient problem. Just clip gradients to be in certain range. adjusted as learning rate annealed, is current learning rate

可调梯度剪辑是处理爆炸/消失梯度问题的有效方式.只需将渐变剪切到一定范围内即可.调整为学习率退火,是目前的学习率

Enables high learning rate (times higher than SRCNN)

实现高学习率(高于SRCNN)

Training takes only 4 hours to achieve SOTA result!

培训只需4小时才能实现SOTA的成果!

## 实验结果

### 了解属性
The deeper, the better

越深越好

For depth D network

深度D网络

the receptive field has size (2D + 1) x (2D + 1)

接收场具有尺寸`(2D + 1) x (2D + 1)`

Deeper depth yields more contextual information

更深入的深度产生更多的情境信息

In addition, exploit high nonlinearity

另外,利用高非线性

Residual vs. non-residual learning

残差与非残差学习

Residual network learns much faster and produces better results than non-residual network

残差网络学习速度快得多,生成比非残差网络更好的结果

One model, Multiple scales

一个模型,多尺度

Interestingly, we observe that training multiple scales boosts the performance for large scales

有趣的是,我们观察到,多尺度的训练可以提升大尺度的性能