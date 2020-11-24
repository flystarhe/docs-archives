title: R入门小抄 | 数据选择
date: 2015-05-20
tags: [R]
---
数据选择`subset`.

<!--more-->
## subset
```
head(airquality)
```

      Ozone Solar.R Wind Temp Month Day
    1    41     190  7.4   67     5   1
    2    36     118  8.0   72     5   2
    3    12     149 12.6   74     5   3
    4    18     313 11.5   62     5   4
    5    NA      NA 14.3   56     5   5
    6    28      NA 14.9   66     5   6

```
head(subset(airquality, Temp > 80, select = c(Ozone, Temp)))
```

       Ozone Temp
    29    45   81
    35    NA   84
    36    NA   85
    38    29   82
    39    NA   87
    40    71   90

```
head(subset(airquality, Day == 1, select = -Temp))
```

        Ozone Solar.R Wind Month Day
    1      41     190  7.4     5   1
    32     NA     286  8.6     6   1
    62    135     269  4.1     7   1
    93     39      83  6.9     8   1
    124    96     167  6.9     9   1

```
head(subset(airquality, select = Ozone:Wind))
```

      Ozone Solar.R Wind
    1    41     190  7.4
    2    36     118  8.0
    3    12     149 12.6
    4    18     313 11.5
    5    NA      NA 14.3
    6    28      NA 14.9
