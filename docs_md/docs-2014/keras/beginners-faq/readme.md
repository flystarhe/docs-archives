title: Keras | For Beginners FAQ
date: 2017-06-08
tags: [深度学习,Keras]
---
Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow或Theano。Keras为支持快速实验而生，能够把你的idea迅速转换为结果。

<!--more-->
## 如何保存Keras模型？[link](http://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/)
我们不推荐使用pickle或cPickle来保存Keras模型。你可以使用`model.save(filepath)`将Keras模型和权重保存在一个HDF5文件中，该文件将包含：

- 模型的结构，以便重构该模型
- 模型的权重
- 训练配置（损失函数，优化器等）
- 优化器的状态，以便于从上次训练中断的地方开始

使用`keras.models.load_model(filepath)`来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译。

例子：
```
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```

如果你只是希望保存模型的结构，而不包含其权重或配置信息，可以使用：
```
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```

当然，你也可以从保存好的json文件或yaml文件中载入模型：
```
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```

如果需要保存模型的权重，可通过下面的代码利用HDF5进行保存：
```
# 已安装了HDF5和Python的h5py库
model.save_weights('my_model_weights.h5')
```

如果你需要在代码中初始化一个完全相同的模型，请使用：
```
model.load_weights('my_model_weights.h5')
```

如果你需要加载权重到不同的网络结构（有些层一样）中，例如`fine-tune`或`transfer-learning`，你可以通过层名字来加载模型：
```
model.load_weights('my_model_weights.h5', by_name=True)
```

例如：
```
"""
假如原模型为：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))
    model.add(Dense(3, name="dense_2"))
    ...
    model.save_weights(fname)
"""
# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
model.add(Dense(10, name="new_dense"))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)
```

## 为什么训练误差比测试误差高很多？[link](http://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/)
Keras的模型有两个模式：训练模式和测试模式。一些正则机制，如Dropout，L1/L2正则项在测试模式下将不被启用。

另外，训练误差是训练数据每个batch的误差的平均。在训练过程中，每个epoch起始时的batch的误差要大一些，而后面的batch的误差要小一些。另一方面，每个epoch结束时计算的测试误差是由模型在epoch结束时的状态决定的，这时候的网络将产生较小的误差。

【Tips】可以通过定义回调函数将每个epoch的训练误差和测试误差并作图，如果训练误差曲线和测试误差曲线之间有很大的空隙，说明你的模型可能有过拟合的问题。当然，这个问题与Keras无关。

## 当验证集的loss不再下降时，如何中断训练？[link](http://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/)
可以定义EarlyStopping来提前终止训练：
```
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
```

## 验证集是如何从训练集中分割出来的？[link](http://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/)
如果在`model.fit`中设置`validation_spilt`的值，则可将数据分为训练集和验证集，例如，设置该值为`0.1`，则训练集的最后`10%`数据将作为验证集，设置其他数字同理。注意，原数据在进行验证集分割前并没有被`shuffle`，所以这里的验证集严格的就是你输入数据最末的`x%`。

## 如何在每个epoch后记录训练/测试的loss和正确率？[link](http://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/)
`model.fit`在运行结束后返回一个`History`对象，其中含有的`history`属性包含了训练过程中损失函数的值以及其他度量指标：
```
hist = model.fit(X, y, validation_split=0.2)
print(hist.history)
```

## 如何利用Keras处理超过机器内存的数据集？[link](http://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/)
可以使用`model.train_on_batch(X,y)`和`model.test_on_batch(X,y)`，请参考[模型](http://keras-cn.readthedocs.io/en/latest/models/sequential/)。

另外，也可以编写一个每次产生一个batch样本的生成器函数，并调用`model.fit_generator(data_generator, samples_per_epoch, nb_epoch)`进行训练。这种方式在Keras代码包的example文件夹下CIFAR10例子里有示范，也可点击[这里](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)在github上浏览。

## 如何在Keras中使用预训练的模型？[link](http://keras-cn.readthedocs.io/en/latest/for_beginners/FAQ/)
我们提供了下面这些图像分类的模型代码及预训练权重：

- VGG16
- VGG19
- ResNet50
- Inception v3

可通过`keras.applications`载入这些模型：
```
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

model = VGG16(weights='imagenet', include_top=True)
```

这些代码的使用示例请参考[Application](http://keras-cn.readthedocs.io/en/latest/other/application/)应用文档。使用这些预训练模型进行特征抽取或`fine-tune`的例子可以参考[此博客](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)。

## 参考资料：
- [Keras中文文档](http://keras-cn.readthedocs.io/en/latest/)
