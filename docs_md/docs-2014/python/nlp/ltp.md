title: LTP
date: 2017-09-18
tags: [Python,NLP]
---
语言技术平台(Language Technology Platform, LTP)是哈工大社会计算与信息检索研究中心历时十年开发的一整套中文语言处理系统.LTP制定了基于XML的语言处理结果表示,并在此基础上提供了一整套自底向上的丰富而且高效的中文语言处理模块(包括词法,句法,语义等6项中文处理核心技术),以及基于动态链接库的应用程序接口,可视化工具,并且能够以网络服务的形式进行使用.

<!--more-->
## 安装
执行命令`pip install pyltp`安装ltp,然后下载ltp需要的数据文件[ltp-data](http://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569),具体使用方法推荐阅读文档[pyltp_doc](http://pyltp.readthedocs.io/zh_CN/latest/)与[ltp_doc](http://ltp.readthedocs.io/zh_CN/latest/).

分词,词性,命名实体识别标注参考[Doc](http://ltp.readthedocs.io/zh_CN/latest/appendix.html).分词外部词典,词性标注外部词典参考[Doc](http://ltp.readthedocs.io/zh_CN/latest/ltptest.html#ltpexlex-reference-label).

## 分词
```python
env_home = '/root/tmps/'
tmp_text = '遥控自动驱动式γ射线后装设备'

from pyltp import Segmentor
segmentor = Segmentor()
## 单纯使用模型
#segmentor.load('ltp_data/cws.model')
## 使用外部词典
segmentor.load_with_lexicon('ltp_data/cws.model','ltp_data/tmp_user_medical.txt')
words = segmentor.segment(tmp_text)
print('\t'.join(words))
segmentor.release()
```

输出为:
```
遥控  自动  驱动式  γ射线  后装  设备
```

## 词性标注
```python
from pyltp import Postagger
postagger = Postagger()
postagger.load('ltp_data/pos.model')
postags = postagger.postag(words)
print('\t'.join(postags))
postagger.release()
```

输出为:
```
v   b   b   n   n   n
```

## 实体识别
```python
from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer()
recognizer.load('ltp_data/ner.model')
netags = recognizer.recognize(words, postags)
print('\t'.join(netags))
recognizer.release()
```

输出为:
```
O   O   O   O   O   O
```
