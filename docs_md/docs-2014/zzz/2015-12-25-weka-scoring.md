title: Using the Weka Scoring Plugin
date: 2015-12-25
tags: [Weka,Kettle,Scoring]
---
Weka scoring 插件是 Kettle transform 部分的一个工具，能够实现 Weka 创建的分类和聚类模型被简单的重用到新的数据。

<!--more-->
### Getting Started
为了使用`Weka scoring`插件，必须先用 Weka 创建模型并导出序列化为 java 对象的模型文件。

1. Starting the Weka Explorer
确保你已经安装了[Weke3.7.10](http://ncu.dl.sourceforge.net/project/weka/weka-3-7/3.7.13/weka-3-7-13.zip)(版本要与 weka-scoring/lib 中 pdm-3.7-ce-3.7.10 匹配)，双击 weka.jar 或通过开始菜单(Windows)启动 weka，点击右侧`Applications/Explorer`。当然，也可以是命令行执行`java -cp $weka_home/weka.jar weka.gui.explorer.Explorer`。

2. Loading Data into the Explorer
装载数据可以是 arrf 和 csv 等文件格式，或者来自数据库。这里装载 arff 格式的数据文件，点击`Open File`选择文件[pendigits.arff](https://github.com/pentaho/pdi-weka-scoring-plugin/tree/master/docs/data)。

3. Building a Classifier
切换到`Classify`选项卡，点击`Classifier`面板的`Choose`按钮，选择学习方案应用到训练数据，这里选择决策树 `trees/J48`。如果对默认`Test options`没有异议，就可以点击`Start`开始训练并评价学习方案。这时，右侧的`Classifier output`报告了学习情况。

4. Exporting the Trained Classifier
你能保存任何你训练的分类器，只需在`Result list`面板中右键对应项`Save model`。训练的模型将会序列化为 java 对象存储到文件中。这里，保存为`J48.model`。

---
### Using the Weka Classifier in Kettle
在 Kettle 中使用训练好的模型到新数据是非常简单的，只需配置`Weka scoring plugin`装载和应用先前创建的模型文件。[pdi-ce-5.0.1.A-stable.zip](http://nchc.dl.sourceforge.net/project/pentaho/Data%20Integration/5.0.1-stable/pdi-ce-5.0.1.A-stable.zip)

1. Preliminaries
首先保证`Weka scoring plugin`正确的安装到`Kettle`。安装非常简单，解压[pdi-wekascoring-plugin-5.0.1-stable-deploy.zip](http://ncu.dl.sourceforge.net/project/pentaho/Data%20Integration/5.0.1-stable/pdi-wekascoring-plugin-5.0.1-stable-deploy.zip)并复制所有文件到`$kettle_home/plugins/steps`。其次，`Kettle`需要相应版本`Weka`库，下载 Weka3.7 并解压文件，复制`weka.jar`文件到`$kettle_home/plugins/steps`。现在启动`Spoon`。

2. A Simple Example
作为`weka scoring`插件应用的简单示范，你将在`weka scoring`中把训练好的模型作用到同样的数据。首先，启动`Spoon`，然后构造`transform`(links a CSV input step to the Weka scoring step)。

然后，配置`CSV input step`，装载[pendigits.csv](https://github.com/pentaho/pdi-weka-scoring-plugin/tree/master/docs/data)。右键`CSV file input`选择`Edit`打开`CSV input`对话框，点击`Browse`选择文件，并分隔符和文件编码是否正确设置。

现在，配置`Weka scoring step`。右键`Weka Scoring`选择`Edit`打开`Weka Scoring`对话框，装载先前创建的`J48.model`。`Fields mapping`标签展示了输入字段和模型属性的配对关系，其中`class`属性映射为`missing (type mis-match)`。(重新打开`CSV input step`配置对话框，`class`字段被识别为`Integer`，改为`String`即可解决。)`Model`标签展示了模型的文本描述。

点击`Spoon`中的`Preview`按钮预览结果，预测的`class(predicted)`字段被追加在末尾。启用`Weka Scoring`对话框`Model file`标签页中的`Output probabilities`选项，结果中将包含判别到各类的可能性。

## 参考资料：
- [pentaho : Using the Weka Scoring Plugin](http://wiki.pentaho.com/display/DATAMINING/Using+the+Weka+Scoring+Plugin)
- [github : Using the Weka Scoring Plugin](https://github.com/pentaho/pdi-weka-scoring-plugin/blob/master/docs/WekaScoring.pdf)