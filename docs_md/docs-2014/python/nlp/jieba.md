title: Jieba
date: 2017-09-19
tags: [Python,NLP]
---
"结巴"支持三种分词模式:精确模式,试图将句子最精确地切开,适合文本分析;全模式,把句子中所有的可以成词的词语都扫描出来,速度非常快,但是不能解决歧义;搜索引擎模式,在精确模式的基础上,对长词再次切分,提高召回率,适合用于搜索引擎分词.

<!--more-->
## 分词

- `jieba.cut`方法接受三个输入参数:需要分词的字符串;`cut_all`参数用来控制是否采用全模式;`HMM`参数用来控制是否使用`HMM`模型
- `jieba.cut_for_search`方法接受两个参数:需要分词的字符串;是否使用`HMM`模型.该方法适合用于搜索引擎构建倒排索引的分词,粒度比较细
- `jieba.cut`以及`jieba.cut_for_search`返回的结构都是一个可迭代的generator
- `jieba.lcut`以及`jieba.lcut_for_search`直接返回`list`
- `jieba.Tokenizer(dictionary=DEFAULT_DICT)`新建自定义分词器,可用于同时使用不同词典.`jieba.dt`为默认分词器,所有全局分词相关函数都是该分词器的映射

### 调整词典

- 使用`add_word(word, freq=None, tag=None)`和`del_word(word)`可在程序中动态修改词典
- 使用`suggest_freq(segment, tune=True)`可调节单个词语的词频,使其能(或不能)被分出来

注意:自动计算的词频在使用`HMM`新词发现功能时可能无效.

"台中"总是被切成"台 中",以及类似情况.解决方法,强制调高词频:

```
jieba.add_word('台中')
# 或 jieba.suggest_freq('台中', True)
```

"今天天气 不错"应该被切成"今天 天气 不错",以及类似情况.解决方法,强制调低词频:

```
jieba.suggest_freq(('今天', '天气'), True)
# 或 jieba.del_word('今天天气')
```

### 延迟加载机制
jieba采用延迟加载,`import jieba`和`jieba.Tokenizer()`不会立即触发词典的加载,一旦有必要才开始加载词典构建前缀字典.如果你想手工初始jieba,也可以手动初始化:

```
import jieba
jieba.initialize()  # 手动初始化
```

在`0.28`之前的版本是不能指定主词典的路径的,有了延迟加载机制后,你可以改变主词典的路径:

```
jieba.set_dictionary('data/dict.txt.big')
```

### 并行分词
将目标文本按行分隔后,把各行文本分配到多个Python进程并行分词,然后归并结果,从而获得分词速度的可观提升.用法：

```
jieba.enable_parallel(4)  # 开启并行分词模式，参数为并行进程数
jieba.disable_parallel()  # 关闭并行分词模式
```

注意:并行分词仅支持默认分词器`jieba.dt`和`jieba.posseg.dt`.

### Tokenize
返回词语在原文的起止位置:

```
result = jieba.tokenize('永和服装饰品有限公司')
for tk in result:
    print("word: %s\t\t start: %d \t\t end: %d" % (tk[0],tk[1],tk[2]))
```

### 实战
默认是精确模式:
```python
import jieba
seg_list = jieba.cut("他来到了网易杭研大厦")
print("/ ".join(seg_list))
```

输出为:
```
他/ 来到/ 了/ 网易/ 杭研/ 大厦
```

全模式:
```python
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("/ ".join(seg_list))
```

输出为:
```
我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
```

精确模式:
```python
seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("/ ".join(seg_list))
```

输出为:
```
我/ 来到/ 北京/ 清华大学
```

搜索引擎模式:
```python
seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
print("/ ".join(seg_list))
```

输出为:
```
小明/ 硕士/ 毕业/ 于/ 中国/ 科学/ 学院/ 科学院/ 中国科学院/ 计算/ 计算所/ ，/ 后/ 在/ 日本/ 京都/ 大学/ 日本京都大学/ 深造
```

### 关闭新词发现
切出了词典中没有的词语,效果不理想?解决方法:
```
jieba.cut('丰田太省了', HMM=False)
jieba.cut('我们中出了一个叛徒', HMM=False)
```

更多问题请点击:https://github.com/fxsjy/jieba/issues?sort=updated&state=closed

### 使用词典
开发者可以指定自己自定义的词典,以便包含jieba词库里没有的词.虽然jieba有新词识别能力,但是自行添加新词可以保证更高的正确率.用法:

```
jieba.load_userdict(file_name)
```

词典格式和`dict.txt`一样:一个词占一行;每一行分三部分:词语,词频(可省略),词性(可省略);用空格隔开,顺序不可颠倒.词频省略时使用自动计算的能保证分出该词的词频.例如:

```
创新办 3 i
云计算 5
凱特琳 nz
台中
```

## 词性标注
`jieba.posseg.POSTokenizer(tokenizer=None)`新建自定义分词器,`tokenizer`参数可指定内部使用的`jieba.Tokenizer`分词器.`jieba.posseg.dt`为默认词性标注分词器.

```
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))
```

输出为:
```
我 r
爱 v
北京 ns
天安门 ns
```

## difflib
该模块提供用于比较序列的类和功能,它可以用于比较文件,并且可以生成各种格式的差异信息,包括HTML和上下文以及统一的差异.要比较目录和文件,请参阅[filecmp](https://docs.python.org/3.5/library/filecmp.html#module-filecmp)模块."看起来"像编辑距离,但这不会产生最小的编辑序列,示例:[Doc](https://docs.python.org/3.5/library/difflib.html)
```python
import difflib

str_1 = 'xabcyb'
str_2 = 'abcbya'
s = difflib.SequenceMatcher(None, str_1, str_2)

print(s.find_longest_match(0,6,0,4))

print(s.get_matching_blocks())

for tag, i1, i2, j1, j2 in s.get_opcodes():
    print ("%7s\t\t a[%d:%d]\t\t (%s)\t\t b[%d:%d]\t\t (%s)" % (tag, i1, i2, str_1[i1:i2], j1, j2, str_2[j1:j2]))
```

输出为:
```
Match(a=1, b=0, size=3)
[Match(a=1, b=0, size=3), Match(a=4, b=4, size=1), Match(a=6, b=6, size=0)]
 delete      a[0:1]      (x)         b[0:0]      ()
  equal      a[1:4]      (abc)       b[0:3]      (abc)
 insert      a[4:4]      ()          b[3:4]      (b)
  equal      a[4:5]      (y)         b[4:5]      (y)
replace      a[5:6]      (b)         b[5:6]      (a)
```

## 参考资料:
- [github: fxsjy/jieba](https://github.com/fxsjy/jieba)