title: PyTorch | Install and VDSR
date: 2017-10-25
tags: [PyTorch,SR]
---
本文首先在Ubuntu16.04中安装PyTorch和Matlab环境.然后,再使用VDSR做超分辨率任务.深度图像重建当下比较受推崇的当属VDSR和DRCN.

<!--more-->
## install pytorch
PyTorch[官网](http://pytorch.org/)提供了多种安装方式,这里选择conda:
```
$ conda install pytorch torchvision cuda80 -c soumith
```

版本情况:

    python                    3.5.3
    pytorch                   0.2.0
    torchvision               0.1.9

由于被墙的原因无法进行选择.如果`conda install`也无法正常工作.则源码安装:
```
$ git clone --recursive https://github.com/pytorch/pytorch
$ cd pytorch
$ export CMAKE_PREFIX_PATH="/home/hejian/anaconda3"
$ conda install numpy pyyaml mkl setuptools cmake cffi
$ conda install -c soumith magma-cuda80 # or magma-cuda75 if CUDA 7.5
$ python setup.py install
```

## install matlab
安装过程参考[ubuntu16.04安装matlab2016b](http://blog.csdn.net/minione_2016/article/details/53313271)完成.下载[matlab 2016b](#)后,首先安装rar工具并解压:
```
$ sudo apt install rar
$ rar x Matlab+2016b+Linux64+Crack.rar
## 注意,命令会解压内容到当前目录
```

注意:解压后Crack文件夹中包含`readme.txt`文件,里面包含密钥;`license_standalone.lic`文件,用于进行软件激活;`/bin/glnxa64/`文件,用于进行matlab安装目录中`bin/glnxa64/`的替换,里面四个文件.

直接挂载就行了,创建个matlab文件夹供挂载,只要挂第一个:
```
$ sudo mkdir /media/matlab
$ sudo mount -o loop R2016b_glnxa64_dvd1.iso /media/matlab/
```

执行`install`开始安装,注意选择`使用文件安装秘钥`:
```
$ cd /media/matlab/
$ cd ..
$ sudo /media/matlab/install
```

默认安装在`/usr/local/`,建议修改.安装到一半,提示拔出dvd1,然后插入dvd2对话框:
```
$ sudo mount -o loop R2016b_glnxa64_dvd2.iso /media/matlab/
```

激活:
```
$ cd /your/path/MATLAB/bin/
$ ./matlab
## 运行matlab,弹出激活对话框
## 选择用不联网的方法进行激活
## 加载license_standalone.lic文件
## 关闭matlab
$ pwd
~/tmp/R2016b/bin/glnxa64
$ sudo cp lib* /your/path/MATLAB/bin/glnxa64/
```

打开新终端,执行`./matlab`启动.卸载镜像:
```
$ sudo umount /media/matlab
```

## VDSR
`Very Deep Convolutional Networks`,关于[VDSR](https://github.com/twtygqyy/pytorch-vdsr),在[漫谈深度学习在超分辨率领域上的应用](http://cvmart.net/community/article/detail/49)有较详细的描述.这里引用PyTorch实现[pytorch-vdsr](https://github.com/twtygqyy/pytorch-vdsr):
```
$ git clone https://github.com/twtygqyy/pytorch-vdsr.git
```

## dataset
数据文件夹中提供一个简单的hdf5格式训练样本,其中包含"数据"和"标签",训练数据用Matlab双三次插值生成,请参考[数据生成代码](https://github.com/twtygqyy/pytorch-vdsr/tree/master/data)来创建训练文件.hdf5格式训练样本:
```
import torch
import h5py

hf = h5py.File('data/train.h5')
data = hf.get('data')
target = hf.get('label')

torch.from_numpy(data[0,:,:,:]).float()
torch.from_numpy(target[0,:,:,:]).float()
```

### 训练
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]

optional arguments:
  -h, --help            Show this help message and exit
  --batchSize           Training batch size
  --nEpochs             Number of epochs to train for
  --lr                  Learning rate. Default=0.01
  --step                Learning rate decay, Default: n=10 epochs
  --cuda                Use cuda
  --resume              Path to checkpoint
  --clip                Clipping Gradients. Default=0.4
  --threads             Number of threads for data loader to use Default=1
  --momentum            Momentum, Default: 0.9
  --weight-decay        Weight decay, Default: 1e-4
  --pretrained          Path to pretrained model (default: none)
```

### 测试
```
usage: test.py [-h] [--cuda] [--model MODEL] [--image IMAGE] [--scale SCALE]

optional arguments:
  -h, --help            Show this help message and exit
  --cuda                Use cuda
  --model               Model path. Default=model/model_epoch_50.pth
  --image               Image name. Default=butterfly_GT
  --scale               Scale factor, Default: 4
```

```
$ cd pytorch-vdsr/
$ test.py --cuda --model model/model_epoch_50.pth --image butterfly_GT --scale 4
```

## 推荐阅读
- [github: ghif/drcn](https://github.com/ghif/drcn)
- [github: twtygqyy/pytorch-vdsr](https://github.com/twtygqyy/pytorch-vdsr)
- [深度学习在图像超分辨率重建中的应用](http://cvmart.net/community/article/detail/11)
- [sub-pixel](https://github.com/pytorch/examples/tree/master/super_resolution)
- [alexjc/neural-enhance](https://github.com/alexjc/neural-enhance)
- [zsdonghao / SRGAN](https://github.com/zsdonghao/SRGAN)

---

请按以下方式生成测试样本,仅在Y通道上评估PSNR:
```
im_gt = imread(img_path);
im_gt = modcrop(im_gt,scale);
im_gt = double(im_gt);
im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
im_l_ycbcr = imresize(im_gt_ycbcr,1/scale,'bicubic');
im_b_ycbcr = imresize(im_l_ycbcr,scale,'bicubic');
im_b_y = im_b_ycbcr(:,:,1) * 255.0;
im_b = ycbcr2rgb(im_b_ycbcr) * 255.0;
```

使用matlab在python中将一个jpg图像转换为numpy数组,使用的代码类似于:
```
im_gt, im_b, im_gt_y, im_b_y = eng.rgb_array(image_name, matlab.double([[scale]]), nargout=4)
im_gt_y = np.array(im_gt_y._data).reshape(im_gt_y.size, order='F')
im_b_y = np.array(im_b_y._data).reshape(im_b_y.size, order='F')
```
