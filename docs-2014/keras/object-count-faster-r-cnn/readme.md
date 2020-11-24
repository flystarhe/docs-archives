title: 目标计数之keras-frcnn
date: 2017-07-25
tags: [Keras,ObjectCount]
---
在机器学习中,精确地计数给定图像或视频帧中的目标实例是很困难的一个问题.很多解决方案被发明出来用以计数行人,汽车和其他目标,但是无一堪称完美.当然,我们正在讨论的是图像处理,所以神经网络不失为解决这一问题的好办法,Faster R-CNN,SSD,YOLOv2.

<!--more-->
## 挑战

找到该问题的合适方案取决于很多因素.除了神经网络图像处理面临的共同挑战之外(比如训练数据的大小,质量等),目标计数问题还有其特殊挑战:

- 计数目标的类型
- 重叠
- 透视
- 检测到的目标的最小尺寸
- 训练和测试速度

如被用以计数高速公路上的汽车或者体育馆前的拥挤人群的方法(其中大多数目标相互重叠,透视使得远距离中存在很小的目标),将大不同于家庭照片中的目标计数方法.同样,这一在单张照片上计数目标的方法也不同于在视频中实时计数目标的方法.

## 简单的需求,简单的方案

在本文中我将尝试使用样本视频(其中多个目标同时可见,但并不过于拥挤)解决街道上的目标计数问题.为了处理拥挤场景或者交通堵塞情况之下的图像从而准确地计数目标实例,建议深研一下该领域内的一篇最新论文:[通过深度学习实现无视角的目标计数](http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro.pdf).通过GitHub上的开源代码可以重现这篇论文中的结果.论文中提及的诸如CCNN和Hydra CNN方法在给定的只有少数几类目标的图像中表现欠佳.因此,不得不另寻他法.机器学习中有一个被称作RCNN(Region based Convolutional Neural Network)的非常有趣的方法,可以识别给定图像中的多个目标和位置.

## 快与更快

有很多方法可以把目标位置寻找和识别的任务结合起来以提升速度和准确度.多年来,我们使用了标准RCNN网络,Fast R-CNN乃至Faster R-CNN取得了长足进展,其中Faster R-CNN被用于解决我们的简单计数问题.Fast RCNN建立在以前的工作上,从而可以使用深度卷积网络高效地分类目标提案(object proposal).相较于RCNN,Fast R-CNN的多项创新使其提升了训练和测试速度以及检测准确度.

在多级管道中(首先检测到目标框,接着进行识别)使用RCNN训练的模型的方法相当慢,且不适用于实时处理.这一方法的主要软肋是速度,在检测目标时,训练和实际测试速度都很慢.通过著名的VGG16,用标准RCNN训练5000张图像用时2.5个GPU-Day,且需要数百GB的存储.测试时使用GPU检测目标每张图像用时47s.这主要是由于在卷积神经网络中为每一个目标提案执行前向传递而不分摊计算造成的.

Fast R-CNN通过引进单步训练算法(可在单个处理阶段分类目标及其空间位置)改善了RCNN,Fast R-CNN中引进的提升有:

- 更高的检测质量
- 通过多任务损失函数实现单一阶段的训练
- 训练可更新所有的网络层
- 功能缓存（feature caching）无需磁盘存储

Faster R-CNN引进了与检测网络共享全图像(full-image)卷积功能的RPN(Region Proposal Network,区域提案网络),使得区域提案几乎没有成本.这一方案的RPN组件告知统一网络检测哪里.对于同一个VGG-16模型,Faster R-CNN在GPU上的帧率为5fps,取得了当前最佳的检测准确度.RPN是一种全卷积网络,可以专门为生成检测提案的任务进行端到端训练,旨在高效地预测纵横比和范围宽广的预测区域提案.

上年,Pinterest使用Faster R-CNN获得了网站视觉搜索能力.下面,我们选择了在被描述的PoC样本视频中检测和计数目标实例.

## 概念证明

为了解决问题,我们将使用Faster R-CNN模型.深度学习框架不止一个,且彼此之间竞争激烈,这使我们处在了有利位置,可以下载最满足我们需求和框架选择的预训练模型.当然你也可以使用提供的训练python脚本自己训练模型,只要记住这可能花费很多天.

Faster R-CNN已存在多个实现,包括Caffe,TensorFlow等等.我们将使用后端为TensorFlow的Keras(v2.0.6),作为原始Keras Fast R-CNN实现的分叉的代码可在[这里](https://github.com/softberries/keras-frcnn)获取.

```python
import keras, tensorflow, PIL, cv2
print(keras.__version__, tensorflow.__version__, PIL.__version__, cv2.__version__, sep="\t\t")
```

    2.0.6       1.2.1       1.13.1      3.4.2       3.1.0

用于测试网络的脚本被修改了,从而它可以处理视频文件,并用合适的数据为被检测的目标(带有概率性)注释每一帧以及被计数目标的摘要.在处理帧时,我也正使用opencv沉重地处理视频和已训练的模型.有一些处理视频的实用方法,比如:
```python
def convert_to_images():
    cam = cv2.VideoCapture(input_video_file)
    counter = 0
    while True:
        flag, frame = cam.read()
        if flag:
            cv2.imwrite(os.path.join(img_path, str(counter) + '.jpg'), frame)
            counter = counter + 1
        else:
            break
        if cv2.waitKey(1) == 27:
            break
            # press esc to quit
    cv2.destroyAllWindows()
```

并从处理的帧中保存视频:
```python
def save_to_video():
    list_files = sorted(get_file_names(output_path), key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    img0 = cv2.imread(os.path.join(output_path, '0.jpg'))
    height, width, layers = img0.shape

    # fourcc = cv2.cv.CV_FOURCC(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
    for f in list_files:
        print("saving..." + f)
        img = cv2.imread(os.path.join(output_path, f))
        videowriter.write(img)
    videowriter.release()
    cv2.destroyAllWindows()
```

尽管目标检测发生在测试中,我们创建了带有被检测目标类别和数字1的元组列表,其稍后将被减少以为特定目标类别计数发生的次数:
```python
for jk in range(new_boxes.shape[0]):
    (x1, y1, x2, y2) = new_boxes[jk, :]

    cv2.rectangle(img_scaled, (x1, y1), (x2, y2), class_to_color[key], 2)

    textLabel = '{}: {}'.format(key, int(100*new_probs[jk]))
    all_dets.append((key, 100*new_probs[jk]))
    all_objects.append((key, 1))
```

以及减少的方法:
```python
def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, sum(item[1] for item in subiter)
```

脚本参数具备相当的自我解释性:

- --input_file,输入视频文件的路径
- --output_file,输出视频文件的路径
- --input_dir,存储已处理帧的输入工作目录的路径
- --output_dir,存储已注释处理帧的输出工作目录的路径
- --frame_rate,在构建视频输出时使用的帧率

## 实验
免费高清影视素材网[videezy](https://www.videezy.com/),保存[22.mp4](https://www.videezy.com/urban/4298-random-cars-driving-by-4k-stock-video)到本地:
```bash
hejian@xlab:/data1/hejian_lab$ source activate py35-tf12-gpu
(py35-tf12-gpu) hejian@xlab:/data1/hejian_lab$ mkdir object_count && cd object_count/
(py35-tf12-gpu) hejian@xlab:/data1/hejian_lab/object_count$ git clone https://github.com/softberries/keras-frcnn.git
(py35-tf12-gpu) hejian@xlab:/data1/hejian_lab/object_count$ mkdir input output
(py35-tf12-gpu) hejian@xlab:/data1/hejian_lab/object_count$ cd keras-frcnn/
(py35-tf12-gpu) hejian@xlab:/data1/hejian_lab/object_count/keras-frcnn$ python test_frcnn_count.py \
--input_file /data1/hejian_lab/object_count/22.mp4 \
--output_file /data1/hejian_lab/object_count/output_22.mp4 \
--input_dir /data1/hejian_lab/object_count/input \
--output_dir /data1/hejian_lab/object_count/output \
--frame_rate=25
```

你可能会遇到各种错误,首先建议你对项目中的Tab全部替换为4个空格.并下载[model_frcnn.hdf5](https://s3-eu-west-1.amazonaws.com/softwaremill-public/model_frcnn.hdf5)到当前目录.然后:

- `keras_frcnn/roi_helpers.py`: `import data_generators`to`from keras_frcnn import data_generators`
- `keras_frcnn/data_generators.py`: `import data_augment`to`from keras_frcnn import data_augment`
- `keras_frcnn/data_generators.py`: `import roi_helpers`to`from keras_frcnn import roi_helpers`
- `keras-frcnn/test_frcnn_count.py`: `config_output_filename = 'config.pickle'`to`config_output_filename = 'keras_frcnn/config.pickle'`
- `keras-frcnn/test_frcnn_count.py`: `with open(config_output_filename, 'r') as f_in:`to`with open(config_output_filename, 'rb') as f_in:`
- `keras-frcnn/test_frcnn_count.py`: `class_mapping = {v: k for k, v in class_mapping.iteritems()}`to`class_mapping = {v: k for k, v in class_mapping.items()}`
- `keras-frcnn/test_frcnn_count.py`: `line 126: np.random.randint(0, 255, 3)`to`np.random.randint(0, 255, 3).tolist()`

确定使用Python2,若要使用Python3,请看这里[keras-frcnn on python 3.5](https://github.com/yhenon/keras-frcnn/pull/70).已经修改妥当的在[这里](http://git.oschina.net/flystarhe/object_count_keras_frcnn).

## 总结

区域深度卷积网络是令人兴奋的工具,可以帮助软件开发者解决很多有趣的问题.本文中展示的方案只是个开始,通过为特定数据集调试网络或者从其他模型中使用迁移学习,我们就可以在检测目标时获得高准确度和速度:

- PoC项目,github地址: https://github.com/softberries/keras-frcnn
- Keras预训练模型: https://s3-eu-west-1.amazonaws.com/softwaremill-public/model_frcnn.hdf5
- Fast R-CNN论文: http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
- Faster R-CNN论文: https://arxiv.org/pdf/1506.01497.pdf
- 本文中使用的视频样本: https://www.videezy.com/

## 参考资料:
- [翻译/神经网络目标计数概述,及Faster R-CNN实现](https://www.jiqizhixin.com/articles/a3e89c5e-dcb9-41f7-b845-60ece072b898)
- [原文/Counting Objects with Faster R-CNN](https://softwaremill.com/counting-objects-with-faster-rcnn/)
- [源码/测试通过](http://git.oschina.net/flystarhe/object_count_keras_frcnn)
