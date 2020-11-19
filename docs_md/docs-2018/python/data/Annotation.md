# Annotation

- [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/)
- [tzutalin/labelImg](https://github.com/tzutalin/labelImg)
- [wkentaro/labelme](https://github.com/wkentaro/labelme)
- [opencv/cvat](https://github.com/opencv/cvat)

## VIA
[VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/)是一款简单独立的手动注释软件，适用于图像，音频和视频。VIA在Web浏览器中运行，不需要任何安装或设置。完整的VIA软件包含于大小小于400千字节的单个HTML页面，该页面在大多数现代Web浏览器中作为离线应用程序运行。

VIA是一个完全基于HTML，Javascript和CSS的开源项目（不依赖于外部库）。VIA由Visual Geometry Group（VGG）开发，并根据BSD-2条款许可发布，该许可允许它对学术项目和商业应用程序都有用。

## LabelImg
[LabelImg](https://github.com/tzutalin/labelImg)是一个图形图像注释工具。它是用Python编写的，并使用Qt作为其图形界面。注释以PASCAL VOC格式保存为XML文件，这是ImageNet使用的格式。此外，它还支持YOLO格式。安装：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple labelImg

labelImg
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

- w, Create a rect box
- d, Next image
- a, Previous image
- Space, Flag the current image as verified

## Labelme
[Labelme](https://github.com/wkentaro/labelme)是一个受`http://labelme.csail.mit.edu`启发的图形图像注释工具。安装：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple labelme
```

启动：
```
labelme dataset -o dataset/flags --flags none,cat,dog --nodata --autosave
labelme dataset -o dataset/labels --labels none,cat,dog --nodata --autosave
```

- Flags are assigned to an entire image. Example
- Labels are assigned to a single polygon. Example

## cvat
[CVAT](https://github.com/opencv/cvat)是用于计算机视觉的免费在线交互式视频和图像注释工具。