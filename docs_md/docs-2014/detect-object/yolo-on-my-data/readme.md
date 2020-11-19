title: Training YOLO on my data
date: 2017-07-27
tags: [YOLO,ObjectDetect]
---
YOLO是一个最先进的实时检测系统,在Titan X处理图像在`40-90`FPS,在`VOC_2007`有78.6%的mAP,在`COCO`的`test-dev`有48.1%的mAP.

<!--more-->
## LabelImg
这款工具是全图形界面,用Python和Qt写的,其标注信息可以直接转化成为XML文件,与PASCAL VOC以及ImageNet用的XML是一样的.在`Ubuntu 16.04`安装[LabelImg](https://github.com/tzutalin/labelImg):
```
git clone https://github.com/tzutalin/labelImg.git && cd labelImg/
sudo apt-get install pyqt5-dev-tools
pip install lxml
make qt5py3
python labelImg.py
python labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

新添加安装方式:
```
pip install labelImg
labelImg
labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```

### Usage
你可以编辑`data/predefined_classes.txt`,自定义类别.步骤:

1. 启动标注工具
2. 点击`Change default saved annotation folder`
3. 点击`Open Dir`
4. 点击`Create RectBox`
5. 点击和释放鼠标左键选择标注区域
6. 使用鼠标右键调整矩形框,复制或移动
7. 注释将保存到指定的文件夹中

相关热键:

- 快捷键`Ctrl+u`,从目录加载所有图像
- 快捷键`Ctrl+r`,修改XML文件保存位置
- 快捷键`Ctrl+s`,保存标注
- 快捷键`Space`,将当前图片标记为已验证
- 快捷键`w`,创建
- 快捷键`d`或按钮`Next Image`,下一张
- 快捷键`a`或按钮`Prev Image`,上一张

### Other
其他同类标注工具:

- [Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)
- [BBox-Label-Tool](https://github.com/puzzledqs/BBox-Label-Tool)
- [ImageLabel](https://github.com/lanbing510/ImageLabel)
- [LabelImg-hej](http://git.oschina.net/flystarhe/labelImg)

## Training YOLO on my data
这里假定要实现一个简单的人脸检测:

- 首先就是数据集的准备,这里使用小工具labelImg
- 模仿VOC的格式建立相应的文件夹

`tree -d my2017`:
```
my2017/
├── Annotations
├── ImageSets
│   └── Main
├── JPEGImages
└── labels
```

`my_2017`为数据集的名字.Annotations存放XML文件.Main中存放`train.txt`,`val.txt`,`test.txt`,只写图片的名字,一行一个.JPEGImages中存放图片.labels中存放由XML生成的txt文件.

修改`scripts/voc_label.py`,将数据集的目录修改为自己的,然后执行`python voc_label.py`,生成labels文件夹,以及文件夹下面的txt标记.

修改`cfg/voc.data`:

- classes,训练的类别数
- train,训练集的txt
- valid,验证集的txt
- names,目标编号对应名称
- backup,weights输出位置

修改`cfg/tiny-yolo.cfg`,最后一个卷积层的filters,最后一个region的classes:
```
filters = num * (classes + coords + 1) = 5*(1+4+1)=30
classes = 1
```

在[这里](https://pjreddie.com/media/files/darknet19_448.conv.23)下载卷积层的权重:
```
curl -O https://pjreddie.com/media/files/darknet19_448.conv.23
```

开始训练:
```
./darknet detector train cfg/voc.data cfg/tiny-yolo.cfg darknet19_448.conv.23
```

测试:
```
./darknet detect cfg/tiny-yolo.cfg your_output_path/tiny-yolo_final.weights data/jiaoshi.jpg
```

## 参考资料:
- [YOLO](http://blog.csdn.net/qq_14845119/article/details/53589282)