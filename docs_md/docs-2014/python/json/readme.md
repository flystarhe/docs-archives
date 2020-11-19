title: Python基础之JSON
date: 2015-05-20
tags: [Python,JSON]
---
JSON作为一种轻量级数据格式，被大量地应用在各种程序环境中。JSON是Javascript的内嵌的标准对象，同时也是MongoDB的表结构存储类型。JSON易于人阅读和编写，同时也易于机器解析和生成。JSON采用独立于语言的文本格式。这些特性使JSON成为理想的数据交换语言。

<!--more-->
## json ex1
一个简单的Python数据结构JSON编码示例。

    # coding=utf-8
    import json
    us_p = [{'name': 'tom', 'age': 22}, {'name': 'anny', 'age': 18}]
    us_j = json.dumps(us_p) # encode
    print("ex1-py:",us_p)
    print("ex1-js:",us_j)
    print("")

## json ex2
Python基础数据类型encode及decode变换前后映射关系。

    # coding=utf-8
    import json
    us_p = (5, 1.5, "fly\" is good", True, None, (1, 2), [1, 2], {"a":1, "b":10})
    us_j = json.dumps(us_p) # encode
    print("ex2-py:",us_p)
    print("ex2-js:",us_j)
    print("ex2-py:",json.loads(us_j)) # decode
    print("")

## json ex3
一个JSON字符串解码为Python对象的示例。

    # coding=utf-8
    import json
    if_j = '[\
    {"name":"nam1","point":{"lat":"39","lng":"116"},"desc":"desc-1"},\
    {"name":"nam2","point":{"lat":"35","lng":"106"},"desc":"desc-2"},\
    {"name":"nam3","point":{"lat":"34","lng":"126"},"desc":"desc-3"}]'
    if_p = json.loads(if_j)
    for p in if_p:
        print("ex3-name:",p["name"])
        print("ex3-point-lat:",p["point"]["lat"])
        print("ex3-point-lng:",p["point"]["lng"])
        print("ex3-desc:",p["desc"])
        print("")

## 参考资料：
- [介绍JSON](http://www.json.org/json-zh.html)
- [JSON入门指南](http://www.ibm.com/developerworks/cn/web/wa-lo-json/index.html)