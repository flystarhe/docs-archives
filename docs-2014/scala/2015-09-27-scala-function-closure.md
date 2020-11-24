title: Scala函数和闭包
date: 2015-09-27
tags: [Scala]
---
当程序变得庞大时，你需要一些方法把它们分割成更小的，更易管理的片段。为了分割控制流，Scala把代码分割成函数。除了作为对象成员函数的方法之外，还有内嵌在函数中的函数，函数文本和函数值。本章带你体会所有Scala中的这些函数的风味。

<!--more-->
## 方法
定义函数最通用的方法是作为某个对象的成员。这种函数被称为方法`method`。

    import scala.io.Source
    object LongLines {
        def processFile(filename: String, width: Int) {
            val source = Source.fromFile(filename)
            for (line <- source.getLines) processLine(filename, width, line)
        }
        private def processLine(filename:String, width:Int, line:String) {
            if (line.length > width) println(filename+": "+line.trim)
        }
    }
    LongLines.processFile(tmp.txt, 30)

## 本地函数
Scala提供了另一种方式：你可以把函数定义在另一个函数中。就好象本地变量那样，这种本地函数仅在包含它的代码块中可见。

    import scala.io.Source
    object LongLines {
        def processFile(filename: String, width: Int) {
            def processLine(line: String) {
                if (line.length > width) print(filename +": "+ line)
            }
            val source = Source.fromFile(filename)
            for (line <- source.getLines) processLine(line)
        } 
    }
    LongLines.processFile(tmp.txt, 30)

## 函数是第一类值
Scala拥有第一类函数`first-classfunction`。你不仅可以定义函数和调用它们，还可以把函数写成没有名字的文本`literal`并把它们像值`value`那样传递。

函数文本被编译进一个类，类在运行期实例化的时候是一个函数值`function value`。因此函数文本和值的区别在于函数文本存在于源代码，而函数值存在于运行期对象。以下是对数执行递增操作的函数文本的简单例子：

    (x: Int) => x + 1

函数值是对象，所以如果你愿意可以把它们存入变量。它们也是函数，所以你可以使用通常的括号函数调用写法调用它们。以下是这两种动作的例子：

    var increase = (x: Int) => x + 1
    increase(10)

如果你想在函数文本中包括超过一个语句，用大括号包住函数体，一行放一个语句，就组成了一个代码块。与方法一样，当函数值被调用时，所有的语句将被执行，而函数的返回值就是最后一行产生的那个表达式。

    increase = (x: Int) => {
        println("hello.")
        x + 1
    }

许多Scala库给你使用它们的机会。例如，所有的集合类都能用到foreach方法。它带一个函数做参数，并对每个元素调用该函数。另一个例子是，集合类型还有filter方法。这个方法选择集合类型里可以通过用户提供的测试的元素。测试是通过函数的使用来提供的。下面是如何用它打印输出所有列表元素的代码：

    val someNumbers = List(-11, -10, -5, 0, 5, 10)
    someNumbers.foreach((x: Int) => println(x))
    someNumbers.filter((x: Int) => x > 0)

## 函数文本的短格式
Scala提供了许多方法去除冗余信息并把函数文本写得更简短。注意留意这些机会，因为它们能让你去掉代码里乱七八糟的东西。一种让函数文本更简短的方式是去除参数类型。第二种去除无用字符的方式是省略类型是被推断的参数之外的括号。前面带过滤器的例子可以写成这样：

    someNumbers.filter((x) => x > 0) //去除参数类型
    someNumbers.filter(x => x > 0) //x两边的括号不是必须的

## 占位符语法
如果想让函数文本更简洁，可以把下划线当做一个或更多参数的占位符`只要每个参数在函数文本内仅出现一次`。你可以把下划线看作表达式里需要被“填入”的“空白”。这个空白在每次函数被调用的时候用函数的参数填入。

    someNumbers.filter(_ > 0)

## 偏应用函数
尽管前面的例子里下划线替代的只是单个参数，你还可以使用一个下划线替换整个参数列表。下面是一个例子：

    someNumbers.foreach(println _) //Scala把它看作下列代码
    someNumbers.foreach(x => println(x))

因此，这个例子中的下划线不是单个参数的占位符。它是整个参数列表的占位符。以这种方式使用下划线时，你就正在写一个偏应用函数`partially applied function`。

    def sum(a: Int, b: Int, c: Int) = a + b + c
    sum(1, 2, 3)
    val a = sum _
    a(1, 2, 3)
    val b = sum(1, _: Int, 3)
    b(2)

## 闭包
函数文本在运行时创建的函数值(对象)被称为闭包`closure`。名称源自于通过“捕获”自由变量的绑定对函数文本执行的“关闭”行动。

    var more = 1
    val addMore = (x: Int) => x + more
    addMore(10)
    more = 9999
    addMore(10)

不带自由变量的函数文本，如`(x: Int) => x + 1`，被称为封闭术语`closed term`，这里术语`term`指的是一小部分源代码。但任何带有自由变量的函数文本，如`(x: Int) => x + more`，都是开放术语`open term`。由于函数值是关闭这个开放术语`(x: Int) => x + more`的行动的最终产物，得到的函数值将包含一个指向捕获的`more`变量的参考，因此被称为闭包。每个闭包都会访问闭包创建时活跃的`more`变量。

## 重复参数
Scala允许你指明函数的最后一个参数可以是重复的。这可以允许客户向函数传入可变长度参数列表。想要标注一个重复参数，在参数的类型之后放一个星号。例如：(这样`echo`可以被零个至多个`String`参数调用)

    def echo(args: String*) =
        for (arg <- args) println(arg)
    echo()
    echo("one")
    echo("hello", "world!")

函数内部，重复参数的类型是声明参数类型的数组。因此`echo`函数里被声明为类型“String*”的`args`的类型实际上是`Array[String]`。然而，如果你有一个合适类型的数组，并尝试把它当作重复参数传入，你会得到一个编译器错误。要实现这个做法，你需要在数组参数后添加一个冒号和一个`_*`符号，像这样：

    val arr = Array("What's", "up", "doc?")
    echo(arr) //<console>:7: error: type mismatch;
    echo(arr: _*)

## 尾递归
像`approximate`这样，在它们最后一个动作调用自己的函数，被称为尾递归`tail recursive`。Scala编译器检测到尾递归就用新值更新函数参数，然后把它替换成一个回到函数开头的跳转。道义上你不应羞于使用递归算法去解决你的问题。递归经常是比基于循环的更优美和简明的方案。如果方案是尾递归，就无须付出任何运行期开销。

    def approximate(guess: Double): Double =
        if (isGoodEnough(guess)) guess
        else approximate(improve(guess))
