title: Jieba
date: 2017-09-17
tags: [Python,NLP]
---
汉字拼音转换工具[python-pinyin](https://github.com/mozillazg/python-pinyin),将汉字转为拼音,可以用于汉字注音,排序,检索.特性:

- 根据词组智能匹配最正确的拼音;
- 支持多音字;
- 简单的繁体支持,注音支持;
- 支持多种不同拼音/注音风格.

<!--more-->
## 安装

    $ pip install pypinyin

详细文档请访问[Doc](http://pypinyin.rtfd.io/).

## 使用
```python
from pypinyin import pinyin, lazy_pinyin
import pypinyin

text = '你好，何剑，重要，三重'

print(pinyin(text))
```

输出为:
```
[['nǐ'], ['hǎo'], ['，'], ['hé'], ['jiàn'], ['，'], ['zhòng'], ['yào'], ['，'], ['sān'], ['chóng']]
```

### 启用多音字模式
```python
print(pinyin(text, heteronym=True))
```

输出为:
```
[['nǐ'], ['hǎo'], ['，'], ['hé', 'hè'], ['jiàn'], ['，'], ['zhòng'], ['yào'], ['，'], ['sān'], ['chóng']]
```

### 设置拼音风格
```python
print(pinyin(text, style=pypinyin.STYLE_NORMAL))
print(pinyin(text, style=pypinyin.STYLE_TONE))
print(pinyin(text, style=pypinyin.STYLE_TONE2))
print(pinyin(text, style=pypinyin.STYLE_TONE3))
```

输出为:
```
[['ni'], ['hao'], ['，'], ['he'], ['jian'], ['，'], ['zhong'], ['yao'], ['，'], ['san'], ['chong']]

[['nǐ'], ['hǎo'], ['，'], ['hé'], ['jiàn'], ['，'], ['zhòng'], ['yào'], ['，'], ['sān'], ['chóng']]

[['ni3'], ['ha3o'], ['，'], ['he2'], ['jia4n'], ['，'], ['zho4ng'], ['ya4o'], ['，'], ['sa1n'], ['cho2ng']]

[['ni3'], ['hao3'], ['，'], ['he2'], ['jian4'], ['，'], ['zhong4'], ['yao4'], ['，'], ['san1'], ['chong2']]
```

### 处理不包含拼音的字符
```python
print(pinyin(text))
print(pinyin(text, errors='ignore')) #处理不含拼音字符：忽略
print(pinyin(text, errors='replace')) #处理不含拼音字符：替换为去掉\u的unicode编码
```

输出为:
```
[['nǐ'], ['hǎo'], ['，'], ['hé'], ['jiàn'], ['，'], ['zhòng'], ['yào'], ['，'], ['sān'], ['chóng']]

[['nǐ'], ['hǎo'], ['hé'], ['jiàn'], ['zhòng'], ['yào'], ['sān'], ['chóng']]

[['nǐ'], ['hǎo'], ['ff0c'], ['hé'], ['jiàn'], ['ff0c'], ['zhòng'], ['yào'], ['ff0c'], ['sān'], ['chóng']]
```

### 不考虑多音字的情况
```python
print(lazy_pinyin(text))
```

输出为;
```
['ni', 'hao', '，', 'he', 'jian', '，', 'zhong', 'yao', '，', 'san', 'chong']
```
