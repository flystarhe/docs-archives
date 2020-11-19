title: Image Processing in Pillow
date: 2017-08-12
tags: [Python,Pillow]
---
Python Imaging Library(PIL)给Python增加了图像处理能力。这个库提供了广泛的文件格式支持，高效的内部展现，以及十分强大的图像处理能力。

<!--more-->
## Python Pillow
准备工作：
```bash
[root@cd1 _bin]# bash Anaconda2-4.3.0-Linux-x86_64.sh
[root@cd1 _bin]# echo $'export PATH=/root/anaconda2/bin:$PATH' >> /etc/profile
[root@cd1 _bin]# source /etc/profile
[root@cd1 _bin]# conda info
[root@cd1 _bin]# conda update conda
[root@cd1 _bin]# conda list | grep -nE "^pil"
[root@cd1 _bin]# pip install pytesseract
```

入门Pillow：[参考](http://pillow-cn.readthedocs.io/zh_CN/latest/guides.html)
```python
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageDraw
from PIL import ImagePath
from PIL import ImageStat
from PIL import ImageOps
import pytesseract
import os, sys

image = Image.open('test.jpg')

```

- size:是一个tuple，表示图像的宽和高，单位为像素
- mode:表示图像的模式，L为灰度图，RGB为真彩色，CMYK为pre-press图像

```python
try:
    with Image.open('test.jpg') as image:
        print(image.format, '%dx%d' % image.size, image.mode)
        image.show()
except IOError:
    pass
```

- save:函数保存图片，除非你指定文件格式，那么文件名中的扩展名用来指定文件格式

```python
infiles = []
for infile in infiles:
    f, e = os.path.splitext(infile)
    outfile = f + '.png'
    if infile != outfile:
        try:
            Image.open(infile).save(outfile)
        except IOError:
            print('cannot convert ' + infile)
```

- 操作图片区域的方法:copy, crop, split, merge, rotate, convert

```python
box = image.copy() #复制图像
box = image.crop((50,10,200,40)) #裁剪图像
r, g, b = image.split() #分离通道
box = Image.merge('RGB', (r, g, b)) #合并通道
out_rota = image.rotate(45) #旋转
out_cmyk = image.convert('CMYK') #模式转换/cmyk
out_gray = image.convert('L') #模式转换/灰度

# tesseract
def tess(image):
    '''
    https://github.com/madmaze/pytesseract/blob/master/src/pytesseract.py
    $ tesseract 1.jpg 1 -l chi_sim -psm 3 makebox
    将得到`1.txt`和`1.box`文件，box文件包含有位置信息
    '''
    print(pytesseract.image_to_string(image))
    print(pytesseract.image_to_string(image,lang='eng+chi_sim'))
    print(pytesseract.image_to_string(image,lang='chi_sim',config='--psm 11'))
    print(pytesseract.image_to_string(image,lang='chi_sim',boxes=True))
```

ImageEnhance模块提供了一些用于图像增强的类:

- ImageEnhance模块的Color类:颜色增强类用于调整图像的颜色均衡。增强因子为0.0将产生黑白图像，为1.0将给出原始图像，为2.0将增大颜色饱和度
- ImageEnhance模块的Brightness类:亮度增强类用于调整图像的亮度。增强因子为0.0将产生黑色图像，为1.0将保持原始图像，为2.0将增大亮度
- ImageEnhance模块的Contrast类:对比度增强类用于调整图像的对比度。增强因子为0.0将产生纯灰色图像，为1.0将保持原始图像，2.0将增大对比度
- ImageEnhance模块的Sharpness类:锐度增强类用于调整图像的锐度。增强因子为0.0将产生模糊图像，为1.0将保持原始图像，为2.0将产生锐化

```python
image_b = ImageEnhance.Brightness(image).enhance(5.0)
tess(image_b)
image_c = ImageEnhance.Color(image).enhance(5.0)
tess(image_c)
image_x = ImageEnhance.Contrast(image).enhance(2.5)
tess(image_x)
image_s = ImageEnhance.Sharpness(image_b).enhance(2.5)
tess(image_s)
```

ImageFilter模块提供了滤波器相关定义:

- ImageFilter.BLUR:为模糊滤波，处理之后的图像会整体变得模糊
- ImageFilter.CONTOUR:为轮廓滤波，将图像中的轮廓信息全部提取出来
- ImageFilter.DETAIL:为细节增强滤波，会使得图像中细节更加明显
- ImageFilter.EDGE_ENHANCE:为边缘增强滤波，突出、加强和改善图像中不同灰度区域之间的边界和轮廓
- ImageFilter.EDGE_ENHANCE_MORE:为深度边缘增强滤波，会使得图像中边缘部分更加明显
- ImageFilter.FIND_EDGES:为寻找边缘信息的滤波，会找出图像中的边缘信息
- ImageFilter.SMOOTH:为平滑滤波，突出图像的宽大区域、低频成分、主干部分或抑制图像噪声和干扰高频成分
- ImageFilter.SMOOTH_MORE:为深度平滑滤波，会使得图像变得更加平滑
- ImageFilter.SHARPEN:为锐化滤波，补偿图像的轮廓，增强图像的边缘及灰度跳变的部分，使图像变得清晰

```python
filter_e1 = image.filter(ImageFilter.EDGE_ENHANCE)
tess(filter_e1)
filter_e2 = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
tess(filter_e2)
filter_s = image.filter(ImageFilter.SHARPEN)
tess(filter_s)
```

ImageDraw模块提供了图像对象的简单2D绘制:

- line(xy,options):在变量xy列表所表示的坐标之间画线，坐标列表可以是任何包含2元组[(x,y),..]或者数字[x,y,..]的序列对象，它至少包括两个坐标，变量options的fill给定线的颜色，变量options的width给定线的宽度
- polygon(xy,options):绘制一个多边形，多边形轮廓由给定坐标之间的直线组成，在最后一个坐标和第一个坐标间增加了一条直线，形成多边形
- rectangle(box,options):绘制一个长边形，它应该包括2个坐标值

```python
draw_line = ImageDraw.Draw(image)
draw_line.line([(0,0),(50,20),(150,30)], fill=(255,0,0), width=5)
image.save('tmp.jpg')
draw_ploy = ImageDraw.Draw(image)
draw_ploy.polygon([(0,0),(50,30),(150,40)], fill=(0,255,0))
image.save('tmp.jpg')
draw_rect = ImageDraw.Draw(image)
draw_rect.rectangle((0,0,200,200), fill=(0,0,255))
image.save('tmp.jpg')

del draw_line
del draw_ploy
del draw_rect
```

ImagePath模块被用于存储和操作二维向量数据:
```python
path = ImagePath.Path((0,0,1,1,20,50,30,20,))
path.compact(distance=2) #去除彼此接近的点压缩path
path.tolist() #将path对象转换为list
path.getbbox() #获取path对象的边界框
```

ImageStat用于计算整个图像或者图像的一个区域的统计数据:
```python
stat = ImageStat.Stat(image)
stat.extrema #每个通道的最大值和最小值
stat.count #每个通道的像素个数
stat.sum #每个通道的像素值之和
stat.mean #每个通道的像素值的平均值
stat.median #每个通道的像素值的中值
stat.rms #每个通道的像素值的均方根值
```

ImageOps.autocontrast:最大图像对比度。这个函数计算一个输入图像的直方图，从这个直方图中去除最亮和最暗的百分之cutoff，然后重新映射图像，以便保留的最暗像素变为黑色，即0，最亮的变为白色，即255
ImageOps.grayscale:将输入图像转换为灰色图像
```python
ops = ImageOps.autocontrast(image, 20)
tess(ops)
ops = ImageOps.grayscale(image)
tess(ops)
```

## 参考资料:
- [Python图像处理库：PIL的ImageFilter](http://blog.csdn.net/icamera0/article/details/50708888)
- [Python图像处理库：Pillow初级教程](http://www.cnblogs.com/wbin91/p/3971079.html)
- [中文/Pillow 官方指南](http://pillow-cn.readthedocs.io/zh_CN/latest/guides.html)