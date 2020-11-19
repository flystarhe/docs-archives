title: R基础之JSON
date: 2015-05-20
tags: [R,JSON]
---
JSON作为一种轻量级数据格式，被大量地应用在各种程序环境中。JSON是Javascript的内嵌的标准对象，同时也是MongoDB的表结构存储类型。JSON易于人阅读和编写，同时也易于机器解析和生成。JSON采用独立于语言的文本格式。这些特性使JSON成为理想的数据交换语言。

<!--more-->
## package: rjson
prepare: install package and require

    if(!require(rjson)){
        install.packages("rjson")
        require(rjson)
    }

## fromJSON: JSON to R

    json_str = paste(readLines("json.file"), collapse="")
    json_dat = rjson::fromJSON(json_str)
    print(json_str)
    print(json_dat)
    ## json.file
    {
        "table1": {
            "time": "130911",
            "data": {
                "code": [
                    "TF1312",
                    "TF1403",
                    "TF1406"
                ],
                "rt_time": [
                    130911,
                    130911,
                    130911
                ]
            }
        },
        "table2": {
            "time": "130911",
            "data": {
                "contract": [
                    "TF1312",
                    "TF1312",
                    "TF1403"
                ],
                "jtid": [
                    99,
                    65,
                    21
                ]
            }
        }
    }

## toJSON: R to JSON

    json_str = rjson::toJSON(json_dat)
    print(json_str) # 转义输出(\")
    writeLines(json_str,"json.file.r")

## 性能测试: C库 vs R库
**4.1. fromJSON**

    print(system.time(y1 <- fromJSON(json_str, method="C")))
    print(system.time(y2 <- fromJSON(json_str, method="R")))
    print(system.time(y3 <- fromJSON(json_str)))

**4.2. toJSON**

    print(system.time(y1 <- toJSON(json_dat, method="C")))
    print(system.time(y2 <- toJSON(json_dat, method="R")))
    print(system.time(y3 <- toJSON(json_dat)))

>rjson的C库比R库会快，fromJSON默认使用的C库的方法

## package: RJSONIO
为解决rjson包序列化大对象慢的问题(不过测试是rjson效果更优)。
prepare: install package and require

    if(!require(RJSONIO)){
        install.packages("RJSONIO")
        require(RJSONIO)
    }

## fromJSON: JSON to R

    json_str = paste(readLines("json.file"), collapse="")
    json_dat = RJSONIO::fromJSON(json_str)
    print(json_str)
    print(json_dat)

## toJSON: R to JSON

    json_str = RJSONIO::toJSON(json_dat)
    print(json_str) # 转义输出(\")
    writeLines(json_str,"json.file.r")

## 自定义JSON实现

    df = data.frame(code = c('TF1312', 'TF1310', 'TF1313'), rt_time = c("152929", "152929", "152929"))
    # 1. toJSON output
    cat(rjson::toJSON(df),"\n")
    cat(RJSONIO::toJSON(df),"\n")
    # 2. we want
    if(!require(plyr)){
        install.packages("plyr")
        require(plyr)
    }
    cat(rjson::toJSON(unname(alply(df, 1, identity))),"\n")
    cat(RJSONIO::toJSON(unname(alply(df, 1, identity))),"\n")

## 参考资料：
- [介绍JSON](http://www.json.org/json-zh.html)
- [R和JSON的傻瓜式编程](http://blog.fens.me/r-json-rjson/)
- [JSON入门指南](http://www.ibm.com/developerworks/cn/web/wa-lo-json/index.html)