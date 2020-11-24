title: R入门小抄 | 数据库操作
date: 2015-05-20
tags: [R]
---
数据库操作.

<!--more-->
## RJDBC
注意，Centos下安装RJDBC需要依赖`yum -y install mysql-devel`:
```r
install.packages("rJava")
install.packages("RJDBC")

setwd("/work_main")

usernm = 'user name'
passwd = 'pass word'
host = 'rm-m5ex1o98v2yq54zr5o.mysql.rds.aliyuncs.com'
port = 3303
dbname = 'calculate'

require(RJDBC)
drv <- JDBC("com.mysql.jdbc.Driver","mysql-connector-java-5.1.40-bin.jar","`")
con <- dbConnect(drv,paste("jdbc:mysql://",host,":",port,"/",dbname,"?characterEncoding=UTF-8",sep=""),usernm,passwd)

dbGetInfo(con)
dbListTables(con)

iris <- data.frame(id=1:5,name=c("苹果","香蕉","梨子","玉米","西瓜"))

if(dbExistsTable(con, "tmp_test_flystar_iris")) {
    dbWriteTable(con, "tmp_test_flystar_iris", iris, overwrite=F, append=T)
} else {
    dbWriteTable(con, "tmp_test_flystar_iris", iris)
}
dbGetQuery(con, "select count(*) from tmp_test_flystar_iris")
dat <- dbReadTable(con, "tmp_test_flystar_iris")

dbDisconnect(con)
```

## RMySQL
注意，Centos下安装RMySQL需要依赖`yum -y install mysql-devel`(win:中文乱码/mac:中文正常/linux:中文正常):
```r
install.packages("RMySQL")
library(RMySQL)

usernm = 'user name'
passwd = 'pass word'
host = 'rm-m5ex1o98v2yq54zr5o.mysql.rds.aliyuncs.com'
port = 3303
dbname = 'calculate'

#help(package="RMySQL")
drv = RMySQL::MySQL()
con <- dbConnect(drv,username=usernm,password=passwd,host=host,port=port,dbname=dbname)
dbSendQuery(con,'SET NAMES utf8')

summary(con)
dbGetInfo(con)
dbListTables(con)

#写数据库表
fruits <- data.frame(id=1:5,name=c("苹果","香蕉","梨子","玉米","西瓜"))
dbWriteTable(con,"tmp_test_flystar_fruits",fruits,overwrite=T,append=F,row.names=F)
dbListTables(con)

#读数据库
data = dbReadTable(con,"tmp_test_flystar_fruits") #中文乱码?

#用SQL语句查询dbGetQuery()
dbGetQuery(con,"SELECT * FROM tmp_test_flystar_fruits limit 3")

#用SQL语句查询dbSendQuery()
res <- dbSendQuery(con,"SELECT *FROM tmp_test_flystar_fruits")
data <- dbFetch(res,n=2) #取前2条数据，n=-1时是获取所有数据
data <- dbFetch(res, n=-1) #取余下所有数据
dbClearResult(res)

dbDisconnect(con)
```

## RPostgreSQL
```r
install.packages("RPostgreSQL")
library(RPostgreSQL)

usernm = 'user name'
passwd = 'pass word'
host = 'rm-m5ex1o98v2yq54zr5o.mysql.rds.aliyuncs.com'
port = 3303
dbname = 'calculate'

#help(package="RPostgreSQL")
drv = dbDriver("PostgreSQL")
con <- dbConnect(drv,username=usernm,password=passwd,host=host,port=port,dbname=dbname)
dbSendQuery(con,'SET NAMES utf8')

summary(con)
dbGetInfo(con)
dbListTables(con)

dbDisconnect(con)
dbUnloadDriver(drv)
```
