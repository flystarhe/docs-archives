title: R画网络图之D3
date: 2017-05-22
tags: [R,网络图,D3]
---
网络图是为了展示数据与数据之间的联系，在生物信息学领域，一般是基因直接的相互作用关系，或者通路之间的联系！

通俗点说，就是我有一些点，它们之间并不是两两相互联系，而是有部分是有连接的，那么我应该如何把这些点画在图片上面呢？因为这些都并没有`X,Y`坐标，只有连接关系，所以我们要根据一个理论来给它们分配坐标，这样就可以把它们画出来了，然后也可以对这些点进行连线，连完线后，就是网络图啦！！！

而给它们分配坐标的理论有点复杂，大概类似于物理里面的万有引力和洛仑磁力相结合来给它们分配一个位置，使得总体的能量最小，就是最稳定状态！而通常这个状态是逼近，而不是精确，所以我们其实对同样的数据可以画出无数个网络图，只需使得网络图合理即可！

<!--more-->
## networkD3
可以看出网络图，就是把所有的点，按照算好的坐标画出来，然后把所有的连线也画出即可！其中算法就是，点的坐标该如何确定？目前支持下列类型的网络图：

- Force directed networks with simpleNetwork and forceNetwork
- Sankey diagrams with sankeyNetwork
- Radial networks with radialNetwork
- Dendro networks with dendroNetwork

来源：[D3 JavaScript Network Graphs from R](http://christophergandrud.github.io/networkD3/)

## simpleNetwork
对于非常基本的网络图形，可以使用`simpleNetwork`。例如：
```{r}
library(networkD3)
src <- c('A','A','A','A','B','B','C','C','D')
target <- c('B','C','D','J','E','F','G','H','I')
net_data <- data.frame(src, target)
simpleNetwork(net_data)
```

## forceNetwork
使用`forceNetwork`更好地控制网络的外观，并绘制更复杂的网络。这是一个例子：
```{r}
# Load data
data(MisLinks)
data(MisNodes)

# Head data
head(MisLinks)
head(MisNodes)

# Plot
forceNetwork(Links = MisLinks, Nodes = MisNodes,
            Source = "source", Target = "target",
            Value = "value", NodeID = "name",
            Group = "group", opacity = 0.8)
```

## sankeyNetwork
也可以使用`sankeyNetwork`创建`Sankey`图。以下是使用下载的JSON数据的示例：
```{r}
# Load energy projection data
URL <- paste0("https://cdn.rawgit.com/christophergandrud/networkD3/",
    "master/JSONdata/energy.json")
Energy <- jsonlite::fromJSON(URL)

# str data
str(Energy)

# Plot
sankeyNetwork(Links = Energy$links, Nodes = Energy$nodes, Source = "source",
            Target = "target", Value = "value", NodeID = "name",
            units = "TWh", fontSize = 12, nodeWidth = 30)
```
