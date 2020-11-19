title: R入门小抄
date: 2015-05-20
tags: [R]
---
R语言作为统计学一门语言，一直在小众领域闪耀着光芒。直到大数据的爆发，R语言变成了一门炙手可热的数据分析的利器。随着越来越多的工程背景的人的加入，R语言的社区在迅速扩大成长。

<!--more-->
## project
```r
options(stringsAsFactors=F)
options(java.parameters='-Xmx4g')
path_data <- "./_data"
path_func <- "./_func"
if(!dir.exists(path_data)) dir.create(path_data)
if(!dir.exists(path_func)) dir.create(path_func)
for(filename in list.files(path_func, pattern="[.][rR]$")){
    filename <- file.path(path_func, filename)
    source(filename, encoding="utf-8")
}
```

## 基础函数
- **help(topic)**获取topic的帮助信息，使用`?topic`可以达到同样效果。`help(package='packagename')`了解R软件包的相关信息。`help.search("topic")`搜索帮助系统。
- **install.packages("name")/remove.packages("name")**安装/卸载包，`update.packages(checkBuilt=TRUE,ask=FALSE)`更新已安装的包。`.packages(all.available=TRUE)`本地安装的包列表。`search()`当前加载的包列表。
- **ls()**显示R环境中的对象名字，`ls.str()`将会展示内存中所有对象的详细信息。`rm(list=ls(all=TRUE))`清除内存中所有对象。`gc()`释放R运行占用的内存。
- **mode()**给出对象的类型。`length()`给出对象的长度。`attributes()`给出对象详细特征，对象当前定义的非内在属性。同类函数还有`str()`和`class()`。
- **sink("out.txt")**重定向输出流到文件，`sink()`则输出到控制台。注意`source()`执行R脚本时对要显示输出的对象需使用`print()`，或者使用`source(file,echo=TRUE)`。`capture.output()`可以将R的输出信息转化为字符或文件。
- **eval()**求解表达式。`parse()`字符串转化为表达式。想在程序中执行动态生成的脚本可以`eval(parse(text="var = x + y"))`。
- **dir.create**新建文件夹。`list.dirs`显示目录下的文件夹。`list.files`显示目录下的文档。`file.exists`判断是否存在，`file.remove`删除，`file.rename`重命名，`file.copy`复制，`file.info`显示信息。
- **read.table("clipboard")/write.table("clipboard")**把数据读/写内存，也就是剪贴板。

## 错误处理
这里简单的介绍如何在R中写出类似java、c、python等主流语言所使用的try-catch机制。

**2.1 错误相关的函数**
warning(...)抛出一个警告；stop(...)拋出一个例外；surpressWarnings(expr)忽略expr中发生的警告；try(expr)尝试执行expr；tryCatch最主流语言例外处理的方法；conditionMessage显示错误讯息。

**2.2 tryCatch的范例**

    tryCatch({
        result = expr;
        }, warning = function(w){
            # 处理警告
        }, error = function(e){
            # 处理错误
        }, finally{
            # 清理
        })

>有时候需要直接回传错误讯息给使用者或是log起来时，可以在tryCatch中使用conditionMessage来提取错误讯息。

## 数据框变换之transform
数据框变换有很多手段，说实话它们的效率都不理想。通过下面示例来感受`transform`带来的速度快感。

    # 方案1 transform - 高效
    d=data.frame(char=letters[1:5],fake_char=as.character(1:5),fac=factor(1:5),char_fac=factor(letters[1:5]),stringsAsFactors=FALSE)
    D=transform(d,fake_char=as.numeric(fake_char),char_fac=as.numeric(char_fac))
    # 性能测试
    x = 1:100000
    x = data.frame(x1=x+0.1,x2=x-0.9,x3=letters[x%%26+1])
    tim=1:15
    for(i in tim) tim[i] = system.time({y=transform(x,x1=as.integer(x1),x2=as.integer(x2),x3=as.integer(x3))})[3]
    tim;mean(tim)
    tim=1:15
    for(i in tim) tim[i] = system.time({z=sapply(x,as.integer)})[3]
    tim;mean(tim)
    str(x);str(y);str(z)

## 点柱图(dot histogram)

    data(iris)
    require(plotrix)
    irnd <- sample(150)
    plen <- iris$Petal.Length[irnd]
    spec <- iris$Species[irnd]
    pwid <- abs(rnorm(150, 0.2))
    ehplot(plen, spec, pch=19, cex=pwid, offset=0.06, col=rainbow(3, alpha=0.6)[as.numeric(spec)], main="cex and col changes", xlab="xlab", ylab="ylab")

## RODBC数据框持久化性能优化
RODBC包提的sqlSave函数保存数据框非常简单，不过性能方面就不敢恭维了。这里我们通过`odbcQuery+insert`来实现高效存储数据框。(相比sqlSave可以节省约90%的时间)

    dbsave <- function(pcon,pdat,pnam){
        sNam=names(pdat)[!sapply(pdat,is.numeric)]
        sFix=paste("insert into ",pnam,"(",paste(names(pdat),collapse=","),")values",sep="")
        if(length(sNam)) sExp=paste(",",sNam,"=paste(\"'\",gsub(\"'\",\"''\",",sNam,"),\"'\",sep=\"\")",sep="",collapse="") else sExp=""
        sExp=paste("apply(transform(pdat",sExp,"),1,paste,sep=\"\",collapse=\",\")",sep="")
        sExp=paste("paste(sFix,paste(\"(\",",sExp,",\")\",sep=\"\",collapse=\",\"),\";\",sep=\"\")",sep="")
        tryCatch({return(odbcQuery(pcon,gsub("NA|'NA'","NULL",eval(parse(text=sExp)))));},error=function(e){})
        return(0)
    }

## dataframe中字符向量的paste困惑
在做RODBC数据框持久化性能优化时发现paste对于含因子的list总是不能正确给出结果，不信你试试这个有趣的脚本：

    x=list(as.factor("hi"),as.factor(c("fly","star")))
    paste(x,sep=",",collapse=";")
    y=data.frame(a=letters[1:6],b=1:6)
    paste(y,sep=",",collapse=";")
    # 方案1 format
    md=data.frame(a=letters[1:6],b=1:6)
    paste(format(md),sep=",",collapse=";")
    # 方案2：apply家族
    md=data.frame(a=letters[1:6],b=1:6)
    apply(md,1,paste,collapse=",")
    apply(md,1,function(p){paste(p,collapse=",")}) # 想玩复杂点你可以这样

## 古怪的R循环
在R巧妙运营apply，lapply，sapply，mapply，vapply，tapply，aggregate，by，split，sweep替代循环是效率运算的关键。
**1. apply**对数组按行或列进行计算，格式为`apply(X, MARGIN, FUN, ...)`。X为一个数组；MARGIN为一个向量，为1表示取行，为2表示取列，为c(1,2)表示行列都运算。
**2. lapply**对列表或向量使用某个函数，格式为`lapply(X, FUN, ...)`。X为list对象，该list的每一个元素都是一个向量；FUN是需要执行的函数。
**3. sapply**是一个用户友好版本的lappy，格式为`sapply(X, FUN, ..., simplify = TRUE, USE.NAMES = TRUE)`。sapply是lapply的特殊形式，`sapply(*, simplify = FALSE, USE.NAMES = FALSE)`和`lapply(*)`的返回值是相同的。
**4. mapply**是函数sapply的变形版，格式为`mapply(FUN, ..., MoreArgs = NULL, SIMPLIFY = TRUE, USE.NAMES = TRUE)`。mapply是sapply的变形版，将FUN依次应用每一个参数的第一个元素、第二个元素、第三个元素上。
**5. vapply**与sapply相类似，但是有返回值得预定义类型，格式为`vapply(X, FUN, FUN.VALUE, ..., USE.NAMES = TRUE)`。vapply类似于sapply函数，但它的返回值有预定义类型，所以使用起来会更安全，有的时候会更快。
**6. tapply**分组统计，格式为`tapply(X, INDEX, FUN = NULL, ..., simplify = TRUE)`。X通常是一个向量；INDEX是一个list对象，且该list中的每一个元素都是与X有同样长度的因子；FUN是需要执行的函数；simplify是逻辑变量，TRUE返回数组，FALSE返回list。
**7. aggregate**计算数据子集的概括统计量。格式为`aggregate(x, by, FUN, ..., simplify = TRUE) | aggregate(formula, data, FUN, ..., subset, na.action = na.omit)`。
**8. by**是tapply应用到数据框的面向对象的封装，格式为`by(data, INDICES, FUN, ..., simplify = TRUE)`。
**9. split**按照因子进行分组，格式为`split(x, f, drop = FALSE, ...)`。
**10. sweep**计算数组的概括统计量，格式为`sweep(x, MARGIN, STATS, FUN = "-", check.margin = TRUE, ...)`。

## 参考资料：
- [R错误处理](http://wush978.github.io/blog/2013/04/04/r-error-handling/)
- [用R进行文件系统管理](http://blog.fens.me/r-file-folder/)