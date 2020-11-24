title: R入门小抄 | 日期时间
date: 2015-05-20
tags: [R]
---
日期时间.

<!--more-->
```r
> d1=Sys.Date()   #日期
> d3=Sys.time()   #时间
> d2=date()       #日期和时间
```

## 日期与数值
```
# 2010-01-01 is 0
library(lubridate)

G_DATE_BEG = days(as.Date('2010-01-01'))@day

string_date2num <- function(str_date, cycle=30){
    tryCatch({
        tmp = days(as.Date(str_date))@day
        return(round((tmp - G_DATE_BEG)/cycle - 0.0001))
    },warning=function(w){
        return(NA)
    },error=function(e){
        return(NA)
    },finally={})
}


string_date2num_inv <- function(num, cycle=30){
    tmp = as.Date('2010-01-01') + num*cycle
    return(paste('(',tmp-cycle/2,',',tmp+cycle/2,']',sep=""))
}
```

```
> myDate=as.Date('2007-08-09')  #"2007-08-09"
> class(myDate)     #Date
> mode(myDate)      #numeric
> as.character(myDate)
```

    # %d    天 (01~31)
    # %a    缩写星期(Mon)
    # %A    星期(Monday)
    # %m    月份(00~12)
    # %b    缩写的月份(Jan)
    # %B    月份(January)
    # %y    年份(07)
    # %Y    年份(2007) 
    # %H    时
    # %M    分
    # %S    秒

```
> birDay=c('01/05/1986','08/11/1976') #"01/05/1986" "08/11/1976"
> dates=as.Date(birDay,'%m/%d/%Y').   #"1986-01-05" "1976-08-11"
> mode(birDay)  #"character"
> mode(dates)   #"numeric"
> format(td,format='%B  %d %Y %s')  #"二月  27 2017 1488124800"
> format(td,format='%A,%a ')        #"星期一,一 "
```

```
> as.integer(Sys.Date())  #自1970年1月1号至今的天数/17224
> as.integer(as.Date('1970-1-1'))  #0
> p=as.POSIXlt(Sys.Date())
> p$year + 1900   #年份需要加1900 -> 2017
> p$mon + 1       #月份需要加1 -> 2
> p$mday          #27
```
