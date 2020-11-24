title: Python3 logging
date: 2017-08-23
tags: [Python]
---
Java中最通用的日志模块莫过于Log4j,在Python中,自带了logging模块,用法与Log4j类似.

<!--more-->
## 基本用法
默认情况下,logging将日志打印到屏幕,日志级别为`WARNING`:
```
import logging

logging.debug('This is debug message')
logging.info('This is info message')
logging.warning('This is warning message')
logging.error('This is error message')
```

屏幕打印:
```
WARNING:root:This is warning message
ERROR:root:This is error message
```

## 日志级别
日志级别大小关系为:

    `CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET`

当然也可以自己定义日志级别.

## Handler
每个logger可以附加多个Handler,常用Handler如下:

- logging.StreamHandler: 日志输出到流,默认是`sys.stderr`
- logging.FileHandler: 和StreamHandler类似,用于向一个文件输出日志信息
- logging.handlers.RotatingFileHandler: 和FileHandler类似,但是它可以管理文件大小.比如,当`chat.log`达到指定的大小之后,把文件改名为`chat.log.1`,如果`chat.log.1`已经存在,则重命名为`chat.log.2`,最后新建`chat.log`继续输出日志
- logging.handlers.TimedRotatingFileHandler: 和RotatingFileHandler类似,不过它没有通过判断文件大小来决定何时重新创建日志文件,而是间隔一定时间就自动创建新的日志文件

## 普通示例
```
#coding=utf-8
import logging
import time

logger = logging.getLogger('name')
logger.setLevel(logging.INFO)

# 输出到屏幕
log_sh = logging.StreamHandler()
log_sh.setLevel(logging.ERROR)
# 输出到文件
log_fh = logging.FileHandler('log_%s'%time.strftime('%H%M%S'))
log_fh.setLevel(logging.WARNING)
# 设置日志格式
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(module)s:%(message)s')
log_sh.setFormatter(formatter)
log_fh.setFormatter(formatter)
# addHandler
logger.addHandler(log_sh)
logger.addHandler(log_fh)

logger.debug('debug message')
logger.info('info message')
logger.warning('warning message')
logger.error('error message')
logger.critical('critical message')
```

屏幕打印:
```
2017-09-22 14:17:16,021-name-ERROR-lab:error message
2017-09-22 14:17:16,021-name-CRITICAL-lab:critical message
```

文件输出:
```
2017-09-22 14:17:16,021-name-WARNING-lab:warning message
2017-09-22 14:17:16,021-name-ERROR-lab:error message
2017-09-22 14:17:16,021-name-CRITICAL-lab:critical message
```

## 基于文件大小切分
```
import time
import logging
import logging.handlers

logger = logging.getLogger('name-size')
logger.setLevel(logging.INFO)

# 输出到文件
# 文件容量1k: maxBytes=1000
# 仅保留最新3个: backupCount=3
filehandler = logging.handlers.RotatingFileHandler('logger.out', maxBytes=1024, backupCount=3)
filehandler.setLevel(logging.INFO)

# addHandler
logger.addHandler(filehandler)

# 测试
for i in range(10000):
    logger.info("file test")
```

## 基于时间间隔切分
```
import time
import logging
import logging.handlers

logger = logging.getLogger('name-time')
logger.setLevel(logging.INFO)

# 输出到文件
# 时间周期单位: when='S','M','H','D'
# 间隔的周期数: interval=1
# 仅保留最新3个: backupCount=3
filehandler = logging.handlers.TimedRotatingFileHandler('logger.out', when='S', interval=1, backupCount=3)
filehandler.setLevel(logging.INFO)

# addHandler
logger.addHandler(filehandler)

# 测试
logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")
time.sleep(1)
logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")
```

## 参考资料:
- [Python日志模块logging详解](https://my.oschina.net/leejun2005/blog/126713)
- [Python3日志木块logging](http://www.cnblogs.com/Devopser/p/6366975.html)
- [Logging HOWTO](https://docs.python.org/3/howto/logging.html)