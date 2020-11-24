title: Scala控制结构
date: 2015-09-27
tags: [Scala]
---
Scala里没有多少内建控制结构。仅有的包括if，while，for，try，match和函数调用。如此之少的理由是，从一开始Scala就包括了函数文本。代之以在基本语法之上一个接一个添加高层级控制结构，Scala把它们汇集在库里。

<!--more-->
有件你会注意到的事情是，几乎所有的Scala的控制结构都会产生某个值。这是函数式语言所采用的方式，程序被看成是计算值的活动，因此程序的控件也应当这么做。换句话说，Scala的if可以产生值。于是Scala持续了这种趋势让for，try和match也产生值。

## if
Scala的if如同许多其它语言中的一样工作。它测试一个状态并据其是否为真，执行两个分支中的一个。

    val filename =
        if (!args.isEmpty) args(0)
        else "default.txt"

## while
Scala的while循环表现的和在其它语言中一样。包括一个状态和循环体，只要状态为真，循环体就一遍遍被执行。

    var a = 12; var b = 9
    while (a != 0) {
        val temp = a; a = b % a; b = temp
    }

Scala也有do-while循环。除了把状态测试从前面移到后面之外，与while循环没有区别。

    var line = ""
    do {
        line = readLine()
        println("Read: " + line)
    } while (line != null)

while和do-while结构被称为“循环”，不是表达式，因为它们不产生有意义的结果，结果的类型是Unit。

## for
你能用for做的最简单的事情就是把一个集合类的所有元素都枚举一遍。

    val filesHere = (new java.io.File(".")).listFiles
    for (file <- filesHere)
        println(file)

for表达式语法对任何种类的集合类都有效，而不只是数组。如果你不想包括被枚举的Range的上边界，可以用until替代to。

    for (i <- 1 to 4)
        println("Iteration " + i)

有些时候你不想枚举一个集合类的全部元素。而是想过滤出一个子集。你可以通过把`过滤器：filter`一个if子句加到for的括号里做到。(如果在发生器中加入超过一个过滤器，if子句必须用分号分隔。)

    val filesHere = (new java.io.File(".")).listFiles
    for (file <- filesHere if file.getName.endsWith(".scala"))
        println(file)

如果加入多个<-子句，你就得到了嵌套的“循环”。如果愿意的话，你可以使用大括号代替小括号环绕发生器和过滤器。使用大括号的一个好处是你可以省略一些使用小括号必须加的分号。

    for {
        file <- filesHere
        if file.getName.endsWith(".scala")
        line <- scala.io.Source.fromFile(file).getLines.toList
        if line.trim.matches(".*for.*")
    } println(file + ": " + line.trim)

请注意前面的代码段中重复出现的表达式line.trim。这不是个可忽略的计算，因此你或许想每次只算一遍。通过用等号(=)把结果绑定到新变量可以做到这点。绑定的变量被当作val引入和使用，不过不用带关键字val。

    for {
        file <- filesHere
        if file.getName.endsWith(".scala")
        line <- scala.io.Source.fromFile(file).getLines.toList
        trimmed = line.trim
        if trimmed.matches(".*for.*")
    } println(file + ": " + trimmed)

你还可以创建一个值去记住每一次的迭代。只要在for表达式之前加上关键字yield。比如，下面的函数鉴别出.scala文件并保存在数组里：`for { 子句 } yield { 循环体 }`

    def scalaFiles =
        for {
            file <- filesHere
            if file.getName.endsWith(".scala")
        } yield file

## try
Scala的异常和许多其它语言的一样。代之用普通方式那样返回一个值，方法可以通过抛出一个异常中止。方法的调用者要么可以捕获并处理这个异常，或者也可以简单地中止掉，并把异常升级到调用者的调用者。异常可以就这么升级，一层层释放调用堆栈，直到某个方法处理了它或没 有剩下其它的方法。

异常的抛出看上去与Java的一模一样。首先创建一个异常对象然后用throw关键字抛出：(如果n不是偶数，那么异常将被抛出。)

    if (n % 2 != 0)
        throw new RuntimeException("n must be even")

用来捕获异常的语法如下。选择catch子句这样的语法的原因是为了与Scala很重要的部分`模式匹配：pattern matching`保持一致。(和其它大多数Scala控制结构一样，try-catch-finally也产生值。)

    import java.io.FileReader
    import java.io.FileNotFoundException
    import java.io.IOException
    try {
        val f = new FileReader("input.txt")
    } catch {
        case ex: FileNotFoundException => // Handle missing file
        case ex: IOException => // Handle other I/O error
    }

## match
Scala的匹配表达式允许你在许多可选项：alternative中做选择，就好象其它语言中的switch语句。

    val friend = firstArg match {
        case "salt" => println("pepper")
        case "chips" => println("salsa")
        case "eggs" => println("bacon")
        case _ => println("huh?")
    }
