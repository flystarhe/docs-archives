title: R入门小抄 | 数据集成
date: 2015-05-20
tags: [R]
---
数据集成`merge`.

<!--more-->
## inner join
```
t1 = data.frame(Id=c(1:6), State=c(rep("北京",3),rep("上海",3)))
t2 = data.frame(Id=c(1,4,6,7), Product=c('IPhone','Vixo','mi','Note2'))

merge(t1, t2, by=c('Id'))
```

      Id State Product
    1  1  北京  IPhone
    2  4  上海    Vixo
    3  6  上海      mi

## full join
```
merge(t1, t2, by=c('Id'), all=T)
```

      Id State Product
    1  1  北京  IPhone
    2  2  北京    <NA>
    3  3  北京    <NA>
    4  4  上海    Vixo
    5  5  上海    <NA>
    6  6  上海      mi
    7  7  <NA>   Note2

## left outer join
```
merge(t1, t2, by=c('Id'), all.x=T) #左边数据都在
```

      Id State Product
    1  1  北京  IPhone
    2  2  北京    <NA>
    3  3  北京    <NA>
    4  4  上海    Vixo
    5  5  上海    <NA>
    6  6  上海      mi

## right outer join
```
merge(t1, t2, by=c('Id'), all.y=T) #右边数据都在
```

      Id State Product
    1  1  北京  IPhone
    2  4  上海    Vixo
    3  6  上海      mi
    4  7  <NA>   Note2

## rbind
```
df1 = data.frame(id=seq(0,by=3,length=5), name=paste('Zhang',seq(0,by=3,length=5)))
df2 = data.frame(id=seq(0,by=4,length=4), name=paste('Zhang',seq(0,by=4,length=4)))

rbind(df1, df2)
```

      id     name
    1  0  Zhang 0
    2  3  Zhang 3
    3  6  Zhang 6
    4  9  Zhang 9
    5 12 Zhang 12
    6  0  Zhang 0
    7  4  Zhang 4
    8  8  Zhang 8
    9 12 Zhang 12

## unique
```
merge(df1, df2, all=T) #去重,不使用by
```

      id     name
    1  0  Zhang 0
    2  3  Zhang 3
    3  4  Zhang 4
    4  6  Zhang 6
    5  8  Zhang 8
    6  9  Zhang 9
    7 12 Zhang 12

## other
```
merge(df1, df2, by=c('id')) #重名的列会被更改显示
```

      id   name.x   name.y
    1  0  Zhang 0  Zhang 0
    2 12 Zhang 12 Zhang 12
