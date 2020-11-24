# Image Segmentation
图像分割（image segmentation）任务的定义是：根据某些规则将图片分成若干个特定的、具有独特性质的区域，并提出感兴趣目标的技术和过程。

目前图像分割任务发展出了以下几个子领域：语义分割（semantic segmentation）、实例分割（instance segmentation）以及今年刚兴起的新领域全景分割（panoptic segmentation）。

而想要理清三个子领域的区别就不得不提到关于图像分割中things和stuff的区别：图像中的内容可以按照是否有固定形状分为things类别和stuff类别，其中，人，车等有固定形状的物体属于things类别（可数名词通常属于things）；天空，草地等没有固定形状的物体属于stuff类别（不可数名词属于stuff）。

语义分割更注重「类别之间的区分」，而实例分割更注重「个体之间的区分」，全景分割可以说是语义分割和实例分割的结合。

目前用于全景分割的常见公开数据集包括：MSCOCO、Vistas、ADE20K和Cityscapes。

- COCO是微软团队公布的可以用来图像recognition、segmentation和captioning的数据集，主要从复杂的日常场景中截取，主要有91个类别，虽然类别比ImageNet少很多，但每一类的图像很多。
- Vistas是全球最大的和最多样化的街景图像数据库，以帮助全球范围内的无人驾驶和自主运输技术。
- ADE20K是一个可用于场景感知、分割和多物体识别等多种任务的数据集。相比于大规模数据集ImageNet和COCO，它的场景更多样化，相比于SUN，它的图像数量更多，对数据的注释也更详细。
- Cityscapes是一个包含50个城市街景的数据集，也是提供无人驾驶环境下的图像分割用的数据集。

对于语义分割和实例分割任务，现在已经有了一些效果很好的模型，为研究者熟知的有语义分割的FCN、Dilated Convolutions、DeepLab、PSPNet等，实例分割的SDS、CFM、FCIS、Mask R-CNN等，而全景分割作为一个2018年刚出现的概念，目前的相关研究仍然屈指可数。

2018年1月，为了找到一种能将stuff和things同时分割开的算法，Facebook人工智能实验室（FAIR）的研究科学家何恺明和他的团队提出了一个新的研究范式：全景分割（Panoptic Segmentation，PS），并定义了新的评价标准。

新的评价指标panoptic quality (PQ) metric，来评价全景分割算法的好坏，PQ的计算方式为：

$$
\begin{aligned}
PQ = \frac{\sum_{(p,g) \in TP} IoU(p,g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}
\end{aligned}
$$

>TP（样本为正，预测结果为正）、FP（样本为负，预测结果为正）和FN（样本为正，预测结果为负），其中p表示预测的segment，ɡ表示ground truth。
