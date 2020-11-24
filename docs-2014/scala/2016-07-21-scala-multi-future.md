title: Scala并行和并发编程
date: 2016-07-21
tags: [Scala,并行计算]
---
Future提供了一套高效非阻塞的方式完成并行操作。Future指的是一类占位符对象，用于指代某些尚未完成计算的结果。一般，由Future的计算结果都是并行执行的，计算完后再使用。以这种方式组织并行任务，便可以写出高效、异步、非阻塞的并行代码。

<!--more-->
## Futures
Future是一种用于指代某个尚未就绪的值的对象。这个值通常是某个计算过程的结果：
+ 若该计算过程尚未完成，我们就说该Future未完成；
+ 若该计算过程正常结束，或中途抛出异常，我们就说该Future已完成。

Future具有一个重要的属性是只能被赋值一次。一旦给定了某个值或某个异常，Future对象就无法再被改写。假设，我们使用某个社交网络假想的API获取某个用户的朋友列表，我们将打开一个新对话(session)，然后发送一个获取特定用户朋友列表的请求：
```
import scala.concurrent._
importExecutionContext.Implicits.global
val session = socialNetwork.createSessionFor("user", credentials)
val f:Future[List[Friend]] = Future {
    session.getFriends()
}
```

上面，首先导入scala.concurrent包。然后，通过一个假想的createSessionFor方法初始化一个向服务器发送请求session变量。这个请求是通过网络发送的，所以可能耗时很长。为了更好的利用CPU，不应该阻塞程序的其他部分，这个计算应该被异步调度。Future方法就是这样做的，它并行地执行指定的计算块。

上面的`import ExecutionContext.Implicits.global`导入默认的全局执行上下文。执行上下文执行提交给他们的任务，你也可把执行上下文看作线程池，这对future方法是必不可少的。因为，它们处理如何和何时执行异步计算。

## 回调函数
我们知道如何开始一个异步计算，但是我们没有演示一旦此结果变得可用后如何使用。我们经常对计算结果感兴趣而不仅仅是它的副作用。在许多Future的实现中，一旦Future的客户端对结果感兴趣，它必须阻塞它自己的计算，并等待直到Future完成，然后才能使用Future的值继续它自己的计算。虽然这在`Scala Future API`中是允许的，但从性能角度来看更好的办法是完全非阻塞，即在Future中注册一个回调。一旦Future完成，就异步调用回调。

注册回调最通常的形式，是使用OnComplete方法，即创建一个`Try[T] => U`类型的回调函数。如果Future成功完成，回调则会应用到`Success[T]`类型的值中，否则应用到`Failure[T]`类型的值中。回到我们社交网络的例子，假设我们想获取最近的帖子并显示在屏幕上：
```
import scala.util.{Success,Failure}
val f:Future[List[String]] = Future {
    session.getRecentPosts
}
f onComplete {
    caseSuccess(posts) => for(post <- posts) println(post)
    caseFailure(t) => println("An error has occured: " + t.getMessage)
}
```

onComplete方法允许客户处理失败或成功的Future结果。对于成功，onSuccess 回调使用如下：
```
val f:Future[List[String]] = Future {
    session.getRecentPosts
}
f onSuccess { 
    case posts => for(post <- posts) println(post)
}
```

onFailure回调只有在future失败，也就是包含一个异常时才会执行。使用如下：
```
val f:Future[List[String]] = Future {
    session.getRecentPosts
}
f onFailure {
    case t => println("An error has occured: " + t.getMessage)
}
```

onComplete、onSuccess和onFailure方法都具有结果类型Unit，这意味着这些回调方法不能被链接。注意，这种设计是为了避免链式调用可能隐含在已注册回调上一个顺序的执行。

## 函数组合&For解构
尽管前文所展示的回调机制已经足够把Future的结果和后继计算结合起来的，但是有时回调机制并不易于使用，且容易造成冗余的代码。可以通过一个例子来说明。假设，我们有一个用于进行货币交易服务的API，想要在有盈利的时候购进一些美元：
```
val rateQuote = Future {
    connection.getCurrentValue(USD)
}
val purchase = rateQuote map { quote =>
    if(isProfitable(quote)) connection.buy(amount, quote)
    else throw new Exception("not profitable")
}
purchase onSuccess {
    case amount => println("Purchased " + amount + " USD")
}
```

我们假设，想把一些美元兑换成法郎。必须为这两种货币报价，然后再在这两个报价的基础上确定交易：
```
val usdQuote = Future { connection.getCurrentValue(USD) }
val chfQuote = Future { connection.getCurrentValue(CHF) }
val purchase = for {
    usd <- usdQuote
    chf <- chfQuote
    if isProfitable(usd, chf)
}yield connection.buy(amount, chf)
purchase onSuccess {
    case amount => println("Purchased "+ amount +" CHF")
}
```

## 阻塞
在Future上阻塞是不鼓励的，因为会出现性能和死锁。回调和组合器才是首选方法，但在某些情况中阻塞也是需要的，并且Futures和Promises的API也支持。在之前的并发交易例子中，在最后有一处用到阻塞来确定是否所有的Futures都已完成。下面是使用的例子：
```
import scala.concurrent._
import scala.concurrent.duration._
def main(args: Array[String]) {
    val rateQuote = Future {
        connection.getCurrentValue(USD)
    }
    val purchase = rateQuote map { quote =>
        if(isProfitable(quote)) connection.buy(amount, quote)
        else throw new Exception("not profitable")
    }
    val lstr = Await.result(purchase, 0 nanos)
}
```

可以调用Await.ready来等待，直到这个Future完成，但获不到结果。

## 参考资料：
- [Scala并行和并发编程-Futures和Promises](http://www.cnblogs.com/liuning8023/p/5186653.html)
- [geotrellis使用（六）Scala并发（并行）编程](http://www.cnblogs.com/shoufengwei/p/5497208.html)
- [JVM并发性: Scala中的异步事件处理](http://www.ibm.com/developerworks/cn/java/j-jvmc4/index.html)