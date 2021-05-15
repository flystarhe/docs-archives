# Keras

## Multi-Label Classification With Keras
[pyimagesearch](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/)

使用我的应用程序,用户将上传他们喜欢的服装照片(例如衬衫,衣服,裤子,鞋子),我的系统将返回类似的项目,并包括他们在线购买衣服的链接.问题是我需要训练分类器将项目分类为不同的类:

- 服装类型:衬衫,连衣裙,裤子,鞋子等
- 颜色:红色,蓝色,绿色,黑色等
- 质地/外观:棉,羊毛,丝绸等

我为三个类别中的每个类别训练了三个独立的CNN,它们的效果非常好.有没有办法将三个CNN合并为一个网络?或者训练一个网络来完成所有三个分类任务?