# GANs and Codes
Keras实现[code](https://github.com/eriklindernoren/Keras-GAN),PyTorch实现[code](https://github.com/eriklindernoren/PyTorch-GAN).[原文@2018-03-01](https://zhuanlan.zhihu.com/p/34139648),[谷歌大脑发布GAN全景图](https://zhuanlan.zhihu.com/p/39792176).

## AC-GAN
带辅助分类器的GAN,全称Auxiliary Classifier GAN.在这类GAN变体中,生成器生成的每张图像,都带有一个类别标签,鉴别器也会同时针对来源和类别标签给出两个概率分布.论文中描述的模型,可以生成符合1000个ImageNet类别的128×128图像.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py

## Adversarial Autoencoder
这种模型简称AAE,是一种概率性自编码器,运用GAN将自编码器的隐藏编码向量和任意先验分布进行匹配来进行变分推断,可以用于半监督分类|分离图像的风格和内容|无监督聚类|降维|数据可视化等方面.在论文中,研究人员给出了用MNIST和多伦多人脸数据集(TFD)训练的模型所生成的样本.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/aae.py

## BiGAN
全称Bidirectional GAN,也就是双向GAN.这种变体能学习反向的映射,也就是将数据投射回隐藏空间.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/bigan/bigan.py

## BGAN
虽然简称和上一类变体只差个`i`,但这两种GAN完全不同.BGAN的全称是boundary-seeking GAN.原版GAN不适用于离散数据,而BGAN用来自鉴别器的估计差异度量来计算生成样本的重要性权重,为训练生成器来提供策略梯度,因此可以用离散数据进行训练.BGAN里生成样本的重要性权重和鉴别器的判定边界紧密相关,因此叫做`寻找边界的GAN`.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/bgan/bgan.py

## CC-GAN
这种模型能用半监督学习的方法,修补图像上缺失的部分.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/ccgan/ccgan.py

## CGAN
条件式生成对抗网络,也就是conditional GAN,其中的生成器和鉴别器都以某种外部信息为条件,比如类别标签或者其他形式的数据.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py

## CycleGAN
这个模型是加州大学伯克利分校的一项研究成果,可以在没有成对训练数据的情况下,实现图像风格的转换.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py

## DCGAN
深度卷积生成对抗网络模型是作为无监督学习的一种方法而提出的,GAN在其中是最大似然率技术的一种替代.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

## DiscoGAN
实现学习发现生成对抗网络的跨域关系.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/discogan/discogan.py

## DualGAN
这种变体能够用两组不同域的无标签图像来训练图像翻译器,架构中的主要GAN学习将图像从域U翻译到域V,而它的对偶GAN学习一个相反的过程,形成一个闭环.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/dualgan/dualgan.py

## GAN
对,就是Ian Goodfellow那个原版GAN.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

## LSGAN
最小平方GAN(Least Squares GAN)的提出,是为了解决GAN无监督学习训练中梯度消失的问题,在鉴别器上使用了最小平方损失函数.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/lsgan/lsgan.py

## Pix2Pix
这个模型大家应该相当熟悉了,用条件对抗网络实现图像到图像的翻译.它和CycleGAN出自同一个伯克利团队,是CGAN的一个应用案例,以整张图像作为CGAN中的条件.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py

## PixelDA
用生成对抗网络实现无监督像素级域自适应.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/pixelda/pixelda.py

## SRGAN
利用生成对抗网络实现单图像超分辨率.将GAN用到了超分辨率任务上,可以将照片扩大4倍.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py

## StackGAN++
尽管生成的敌对网络(GANs)在各种任务中已经取得了显著的成功,但它们仍然在生成高质量图像方面面临挑战.本文提出了一种堆叠的生成对抗网络(StackGAN),目标是生成高分辨率的现实图像.

首先,本文提出了一个包含两阶段的生成对抗网络架构`StackGAN-v1`用于文本-图像合成.根据给定的文字描述,GAN在第一阶段描绘出了物体的原始形状和颜色,产生了低分辨率的图像.在第二阶段,GAN将第一阶段的低分辨率图像和文字描述作为输入,并以逼真的细节生成高分辨率的图像.

其次,提出了一种多阶段的生成对抗性网络架构,即`StackGAN-v2`,用于有条件和无条件的生成任务.提出的`StackGAN-v2`由多个树状结构的生成器和判别器组成.树的不同分支可以生成对应于同一场景的多个尺寸的图像.通过对多个分布的联合逼近,`StackGAN-v2`显示了比`StackGAN-v1`更稳定的训练结果.大量的实验证明,在生成高清图像时,文章提出的堆叠的生成对抗网络比其他现阶段表现优异的算法更具优势.

- paper: https://arxiv.org/abs/1710.10916v3

## WGAN
这种变体全称Wasserstein GAN,在学习分布上使用了Wasserstein距离,也叫Earth-Mover距离.新模型提高了学习的稳定性,消除了模型崩溃等问题,并给出了在debug或搜索超参数时有参考意义的学习曲线.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py

## WGAN GP
改进了Wasserstein GAN的训练.

- code: https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py

## 推荐阅读：
- [OpenAI - 生成模型](https://blog.openai.com/generative-models/)