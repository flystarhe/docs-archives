title: R常用图形举例
date: 2015-05-21
tags: [R,ggplot2]
---
R以快速更新的算法，灵活的编程，广泛的扩展，绚丽的图形著称。随着越来越多的工程背景的人的加入，R语言的社区在迅速扩大成长。

<!--more-->
## use graphics
    #stem
    stem(islands, scale=1, width=80)
    #dotchart
    dotchart(VADeaths, main="Death Rates in Virginia - 1940")
    #boxplot
    boxplot(count ~ spray, data=InsectSprays, col="bisque")
    boxplot(data.frame(col1=runif(100,3,9), col2=runif(100,1,7)))
    #hist
    hist(sqrt(islands), breaks=12, col="lightblue", border="pink")
    #qqnorm
    y = rt(200, df=5); qqnorm(y); qqline(y, col = 2)
    #contour
    contour(outer(-6:60, -6:60), method="edge", vfont=c("sans serif", "plain"))
    #pairs
    pairs(iris[-5], main='i', pch=21, bg=c("red","green","blue")[iris$Species])
    #coplot
    coplot(lat ~ long | depth, data = quakes)

## use ggplot2
    install.packages('ggplot2')
    require('ggplot2')
    str(mpg) #data
    str(diamonds) #data

## 散点图
    p = ggplot(data=mpg, mapping=aes(x=cty, y=hwy))
    p +
     geom_point(aes(colour=class, size=displ), alpha=0.5, position='jitter') +
     stat_smooth() +
     scale_size_continuous(range=c(4,10)) +
     facet_wrap(~year, ncol=1) +
     ggtitle('汽车油耗与型号') +
     labs(y='每加仑高速公路行驶距离', x='每加仑城市公路行驶距离') +
     guides(size=guide_legend(title='排量'), colour=guide_legend(title='车型', override.aes=list(size=5)))

### 散点图1 普通模式
    p = ggplot(data=mpg, mapping=aes(x=cty, y=hwy, colour=factor(year)))
    p + geom_point() #普通散点图
    p + geom_point() + stat_smooth() #带平滑线的散点图

### 散点图2 整体平滑
    p = ggplot(data=mpg, mapping=aes(x=cty, y=hwy))
    p + geom_point(aes(colour=factor(year))) #普通散点图
    p + geom_point(aes(colour=factor(year))) + stat_smooth() #带平滑线的散点图

### 散点图3 气泡效果
    p = ggplot(data=mpg, mapping=aes(x=cty, y=hwy, colour=factor(year)))
    p + geom_point(aes(size=displ)) #气泡图
    p + geom_point(aes(size=displ)) + scale_size_continuous(range=c(4,10)) #设置气泡大小

### 散点图4 大数据可视化
    p = ggplot(data=mpg, mapping=aes(x=cty, y=hwy, colour=factor(year)))
    p + geom_point(alpha=0.5) #半透明效果 --针对大数据可视化
    p + geom_point(alpha=0.5, position='jitter') #抖动效果 --针对大数据可视化

### 散点图5 其它绘图设置
    p = ggplot(data=mpg, mapping=aes(x=cty, y=hwy, colour=factor(year)))
    p + coord_cartesian(xlim=c(15,25), ylim=c(15,40)) #窗口
    p + stat_smooth() + facet_wrap(~year, ncol=1) #分面

## 直方图
    p = ggplot(data=mpg, mapping=aes(x=hwy, colour=factor(year)))
    p +
     geom_histogram(aes(fill=factor(year), y=..density..), alpha=0.5) +
     stat_density(geom='line', position='identity', size=1) +
     facet_wrap(~year, ncol=1)

### 直方图1
    p = ggplot(data=mpg, mapping=aes(x=class))
    p + geom_bar(aes(fill=class)) #填充颜色
    p + geom_bar(aes(fill=class)) + facet_wrap(~year) #分面

### 直方图2
    p = ggplot(data=mpg, mapping=aes(x=class, fill=factor(year)))
    p + geom_bar(position='identity') #普通柱形图
    p + geom_bar(position='dodge') #簇状柱形图
    p + geom_bar(position='stack') #堆叠柱形图
    p + geom_bar(position='fill') #百分比堆叠柱形图

## 饼形图
    p = ggplot(data=mpg, mapping=aes(x=factor(1), fill=factor(class)))
    p + geom_bar(width=1)
    p + geom_bar(width=1) + coord_polar(theta='y')

## 箱形图
    p = ggplot(data=mpg, mapping=aes(x=class, y=hwy, fill=class))
    p + geom_boxplot() + geom_jitter(shape=21)
    p + geom_violin(alpha=0.5, width=0.9) + geom_jitter(shape=21)

## 观察密集散点之二维直方图
    p = ggplot(data=diamonds, mapping=aes(x=carat, y=price))
    p + geom_point()
    p + stat_bin2d(bins=60)

## 观察密集散点之二维密度图
    p = ggplot(data=diamonds, mapping=aes(x=carat, y=price))
    p + stat_density2d(aes(fill=..level..), geom='polygon')

### 实例1：玫瑰图
    #随机生成100次风向并汇集到16个区间
    dir = cut_interval(runif(100, 0, 360), n=16)
    #随机生成100次风速并划分成04种强度
    mag = cut_interval(rgamma(100,15), n=4)
    sample = data.frame(dir=dir, mag=mag)
    #将风向映射到x轴，频数映射到y轴，强度映射到填充色
    p = ggplot(data=sample, mapping=aes(x=dir, y=..count.., fill=mag))
    p + geom_bar() + coord_polar()

### 实例2：时间序列
    require(quantmod)
    getSymbols('^SSEC', src='yahoo', from='1997-01-01')
    close = Cl(SSEC)
    time = index(close)
    value = as.vector(close)
    yrng = range(value)
    xrng = range(time)
    data = data.frame(start=as.Date(c('1997-01-01','2003-01-01')), end=as.Date(c('2002-12-30','2012-01-20')), core=c('jiang','hu'))
    timepoint = as.Date(c('1999-07-02','2001-07-26','2005-04-29','2008-01-10','2010-03-31'))
    events = c('证券法实施','国有股减持','股权分置改革','次贷危机爆发','融资融券试点')
    data2 = data.frame(timepoint, events, stock=value[time %in% timepoint])
    p = ggplot(data=data.frame(time,value), mapping=aes(time,value))
    p + geom_line(size=1, colour='turquoise4') +
     geom_rect(alpha=0.2, aes(NULL, NULL, xmin=start, xmax=end, fill=core), ymin=yrng[1], ymax=yrng[2], data=data) +
     scale_fill_manual(values=c('blue','red')) +
     geom_text(aes(timepoint, stock, label=events), data=data2, vjust=-2, size=5) +
     geom_point(aes(timepoint, stock), data=data2, size=5, colour='red', alpha=0.5)
