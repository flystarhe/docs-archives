title: Scala入门小抄
date: 2015-09-23
tags: [Scala]
---
Scala允许你用指令式风格编程，但是鼓励你采用函数式的风格。如果代码包含了任何var变量，那它大概就是指令式的风格。如果代码仅仅包含val，那它大概是函数式的风格。

<!--more-->
函数式的代码比指令式的代码更简洁，明白，也更少机会犯错。Scala鼓励函数式风格的原因，实际上也就是因为函数式风格可以帮助你写出更易读懂，更容易测试，更不容易犯错的代码。不过请牢记在心：不管是var还是副作用都不是天生邪恶的。

## Function
Scala里方法参数的一个重要特征是它们都是val，不是var。如果没有发现任何显式的返回语句，Scala方法将返回方法中最后一个计算得到的值。假如某个方法仅计算单个结果表达式，则可以去掉大括号。如果结果表达式很短，甚至可以把它放在def同一行里。

    def max(x: Int, y: Int): Int = {
        if(x > y) x else y
    }
    println("max(4, 5) = %s".format(max(4, 5)))

令人困惑的地方是当你去掉方法体前面的等号时，它的结果类型将注定是Unit。如果方法的最后结果是String，但方法的结果类型被声明为Unit，那么String将被转变为Unit并失去它的值。

    def dis(s: String){ s.length + " | " + s }
    println(dis("hello.")) //打印的是“()”

## Loop

    val args = Array("zero", "one", "two\n")
    // 示例1
    var i = 0
    while(i < args.length){
        print(args(i))
        i += 1
    }
    // 示例2
    args.foreach(arg => print(arg))
    // 示例3
    for(arg <- args) print(arg)
    // 示例4
    for(i <- 0 to 2) print(args(i))

## Array
Scala里的数组是通过把索引放在圆括号里面访问的，而不是像Java那样放在方括号里。

    // 示例1
    val greetStrings = new Array[String](3)
    greetStrings(0) = "zero"
    greetStrings(1) = "one"
    greetStrings(2) = "two\n"
    // 示例2
    val numNames = Array("zero", "one", "two\n")

## List
Scala的List是设计给函数式风格的编程用的。Scala的List总是不可变的。当你在一个List上调用方法时，似乎这个名字指代的List看上去被改变了，而实际上它只是用新的值创建了一个List并返回。

    List() //或Nil 空List
    List("Cool", "tools", "rule") //创建带有三个值"Cool"，"tools"和"rule"的新List[String]
    val thrill = "Will"::"fill"::"until\n"::Nil //创建带有三个值"Will"，"fill"和"until"的新List[String]
    List("a", "b") ::: List("c", "d") //叠加两个列表（返回带"a"，"b"，"c"和"d"的新List[String]）
    thrill(2) //返回在thrill列表上索引为2（基于0）的元素（返回"until"）
    thrill.count(s => s.length == 4) //计算长度为4的String元素个数（返回2）
    thrill.drop(2) //返回去掉前2个元素的thrill列表（返回List("until")）
    thrill.dropRight(2) //返回去掉后2个元素的thrill列表（返回List("Will")）
    thrill.exists(s => s == "until") //判断是否有值为"until"的字串元素在thrill里（返回true）
    thrill.filter(s => s.length == 4) //依次返回所有长度为4的元素组成的列表（返回List("Will", "fill")）
    thrill.forall(s => s.endsWith("1")) //辨别是否thrill列表里所有元素都以"l"结尾（返回true）
    thrill.foreach(s => print(s)) //对thrill列表每个字串执行print语句 或thrill.foreach(print)
    thrill.head //返回thrill列表的第一个元素（返回"Will"）
    thrill.init //返回thrill列表除最后一个以外其他元素组成的列表（返回List("Will", "fill")）
    thrill.isEmpty //说明thrill列表是否为空（返回false）
    thrill.last //返回thrill列表的最后一个元素（返回"until"）
    thrill.length //返回thrill列表的元素数量（返回3）
    thrill.map(s => s + "y") //返回由thrill列表里每一个String元素都加了"y"构成的列表（返回List("Willy", "filly", "untily")）
    thrill.mkString(", ") //用列表的元素创建字串（返回"will, fill, until"）
    thrill.reverse //返回含有thrill列表的逆序元素的列表（返回List("until", "fill", "Will")）
    thrill.tail //返回除掉第一个元素的thrill列表（返回List("fill", "until")）

## Tuple
元组与列表一样，元组也是不可变的，但与列表不同，元组可以包含 **不同类型** 的元素。实例化一个装有一些对象的新元组，只要把这些对象放在括号里，并用逗号分隔即可。一旦你已经实例化了一个元组，你可以用点号，下划线和一个基于1的元素索引访问它。

    val pair = (99, "Luftballons")
    println(pair._1)
    println(pair._2)

## Set & Map

    import scala.collection.mutable.Set
    val jetSet = Set("Boeing", "Airbus")
    jetSet += "Lear"
    println(jetSet)
    import scala.collection.mutable.Map
    val treasureMap = Map[Int, String]()
    treasureMap += (1 -> "Go to island.")
    treasureMap += (2 -> "Find big X on ground.")
    treasureMap += (3 -> "Dig.")
    println(treasureMap)

## File
处理琐碎的，每日工作的脚本经常需要处理文件。接下来，将演示一个从文件中读行记录，并把行中字符个数前置到每一行，打印输出的脚本。

    import scala.io.Source
    for(line <- Source.fromFile("scala.scala").getLines)
        println(line.length + " " + line)

尽管当前形式的脚本打印出了所需的信息，你或许希望能让数字右序排列，并加上管道符号。

    val lines = Source.fromFile("scala.scala").getLines.toList
    def widthOfLength(s: String): Int = s.length.toString.length
    var maxWidth = 0
    // 方案1
    for(line <- lines)
        maxWidth = maxWidth.max(widthOfLength(line))
    // 方案2
    val longestLine = lines.reduceLeft((a, b) => if(a.length > b.length) a else b)
    maxWidth = widthOfLength(longestLine)
    // 输出
    for(line <- lines){
        val numSpaces = maxWidth - widthOfLength(line)
        val padding = " " * numSpaces
        println(padding + line.length + " | " + line)
    }

reduceLeft方法把传入的方法应用于lines的前两个元素，然后再应用于第一次应用的结果和lines接下去的一个元素，等等，直至整个列表。

## Scala程序
要执行Scala程序，你一定要提供一个有main方法（仅带一个参数，Array[String]，且结果类型为Unit）的孤立单例对象名。任何拥有合适签名的main方法的单例对象都可以用来作为程序的入口点。

    // 示例1
    object Test1{
        def main(args: Array[String]){
            println("hello scala.")
        }
    }
    // 示例2
    object Test2{
        def main(args: Array[String]): Unit = {
            println("hello scala.")
        }
    }

把`ScalaFunctions.scala`和`Scala.scala`放在同一目录。命令行执行：`scalac ScalaFunctions.scala Scala.scala`和`scala Scala`。屏幕打印`12 | hello scala.`。

    // 文件Scala.scala
    import ScalaFunctions.tail
    object Scala{
        def main(args: Array[String]): Unit = {
            println(tail("hello scala."))
        }
    }
    // 文件ScalaFunctions.scala
    class ScalaFunctions{
        val name = "ScalaFunctions"
    }
    object ScalaFunctions{
        def tail(s: String): String = s.length + " | " + s
    }

## Scala脚本
object mode:

    import scala.io.Source
    object LongLines{
        def processFile(filename: String, width: Int){
            val source = Source.fromFile(filename)
            for(line <- source.getLines)
                processLine(filename, width, line)
        }
        private def processLine(filename:String, width:Int, line:String){
          if(line.length > width)
            println(filename + ":[" + line.length + "] " + line.trim)
        }
    }
    LongLines.processFile("scala.txt", 2)

function mode:

    import scala.io.Source
    def processFile(filename: String, width: Int){
        val source = Source.fromFile(filename)
        for(line <- source.getLines)
            processLine(filename, width, line)
    }
    def processLine(filename:String, width:Int, line:String){
        if(line.length > width)
            println(filename + ":[" + line.length + "] " + line.trim)
    }
    processFile("scala.txt", 2)

closure mode:

    import scala.io.Source
    def processFile(filename: String, width: Int){
        def processLine(filename:String, width:Int, line:String){
            if(line.length > width)
                println(filename + ":[" + line.length + "] " + line.trim)
        }
        val source = Source.fromFile(filename)
        for(line <- source.getLines){
            processLine(filename, width, line)
        }
    }
    processFile("scala.txt", 2)

## 参考资料：
- [Scala 教程](http://www.runoob.com/scala/scala-tutorial.html)