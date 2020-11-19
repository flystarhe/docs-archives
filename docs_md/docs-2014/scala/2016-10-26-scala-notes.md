title: Scala进阶笔记
date: 2016-10-26
tags: [Scala]
---
Scala是一门多范式（multi-paradigm）的编程语言，设计初衷是要集成面向对象编程和函数式编程的各种特性。Scala运行在Java虚拟机上，并兼容现有的Java程序。

<!--more-->
## 空对象
1. null是`Null`的实例，类似Java中的`null`。
2. Nothing是`trait`，定义为`final trait Nothing extends Any`。`Nothing`是所有类型的子类型，没有实例。
3. Null是`trait`，定义为`final trait Null extends AnyRef`。`Null`是所有引用类型的子类型，唯一的实例是`null`。
4. Nil是`case object`，定义为`case object Nil extends List[Nothing]`，代表一个空`list`。由于Scala中的`List`是协变的，因此无论`T`是何种类型，`Nil`都是`List[T]`的实例。
5. None是`case object`，定义为`case object None extends Option[Nothing]`，代表不存在的值。Option有两个实例，`None`和`Some`。
6. Unit是`class`，定义为`abstract final class Unit extends AnyVal`。跟Java中的`void`相当，`Unit`唯一的实例是`()`。

## 字符串/插值器
像数组，字符串不是直接的序列，但是他们可以转换为序列，并且也支持所有的序列操作，这里有些例子让你可以理解在字符串上操作：
```
scala> val str = "hello"
str: java.lang.String = hello
scala> str.reverse
res6: String = olleh
scala> str.map(_.toUpper)
res7: String = HELLO
scala> str drop 3
res8: String = lo
scala> str slice (1, 4)
res9: String = ell
scala> val s: Seq[Char] = str
s: Seq[Char] = WrappedString(h, e, l, l, o)
```

`s`插值器，可以直接在字符串中使用`变量`和`表达式`。如下：
```
val name = "James"
println(s"Hello, $name. 1+1=${1+1}?")
```

`f`插值器，功能相似于其他语言中的`printf`函数。当使用`f`插值器的时候，所有的变量引用都应当后跟一个`printf-style`格式串，如`%d`。看下面这个例子：
```
val height = 1.9d
val name = "James"
println(f"$name%s is ${height%2.2f} meters tall")
```

`raw`插值器。除了对字面值中的字符不做编码外，`raw`插值器与`s`插值器在功能上是相同的。如下是个被处理过的字符串：
```
raw"a\nb"  // 当不想输入\n被转换为回车的时候是非常实用的
```

## 数组
在Scala中，数组是一种特殊的collection。一方面，Scala数组与Java数组是一一对应的。即Scala数组`Array[Int]`可看作Java的`Int[]`。但Scala数组比Java数组提供了更多内容。首先，Scala数组是一种泛型。即可以定义一个`Array[T]`，`T`可以是一种类型参数或抽象类型。其次，Scala数组与Scala序列是兼容的，在需要`Seq[T]`的地方可由`Array[T]`代替。最后，Scala数组支持所有的序列操作。这里有个实际的例子：
```
scala> val a1 = Array(1, 2, 3)
a1: Array[Int] = Array(1, 2, 3)
scala> val a2 = a1 map (_ * 3)
a2: Array[Int] = Array(3, 6, 9)
scala> val a3 = a2 filter (_ % 2 != 0)
a3: Array[Int] = Array(3, 9)
scala> a3.reverse
res1: Array[Int] = Array(9, 3)
```

数组与序列是兼容的，因为数组可以隐式转换为WrappedArray。反之可以使用`toArray`方法将WrappedArray转换为数组。最后一行表明，隐式转换与toArray方法作用相互抵消。
```
scala> val seq: Seq[Int] = a1
seq: Seq[Int] = WrappedArray(1, 2, 3)
scala> val a4: Array[Int] = seq.toArray
a4: Array[Int] = Array(1, 2, 3)
scala> a1 eq a4
res2: Boolean = true
```

数组还有另外一种隐式转换，不需要将数组转换成序列，而是简单地把所有序列的方法“添加”给数组。“添加”其实是将数组封装到一个ArrayOps类型的对象中，后者支持所有序列的方法。ArrayOps对象的生命周期通常很短暂，不调用序列方法的时候基本不会用到，其内存也可以回收。
```
scala> val seq: Seq[Int] = a1
seq: Seq[Int] = WrappedArray(1, 2, 3)
scala> seq.reverse
res2: Seq[Int] = WrappedArray(3, 2, 1)
scala> val ops: collection.mutable.ArrayOps[Int] = a1
ops: scala.collection.mutable.ArrayOps[Int] = [I(1, 2, 3)
scala> ops.reverse
res3: Array[Int] = Array(3, 2, 1)
```

## 容器类
Scala中提供了多种具体的不可变集类供你选择，这些类`maps, sets, sequences`实现的接口`traits`不同，比如是否能够是无限`infinite`，各种操作的速度也不一样：

- **List（列表）：**是一种有限的不可变序列式。提供了常数时间的访问列表头元素和列表尾的操作，并且提供了常数时间的构造新链表的操作，该操作将一个新的元素插入到列表的头部。其他许多操作则和列表的长度成线性关系。
- **Stream（流）：**与List很相似，只不过其中的每一个元素都经过了一些简单的计算处理。也正是因为如此，stream结构可以无限长。只有那些被要求的元素才会经过计算处理，stream被特别定义为懒惰计算。
- **Vector（向量）：**向量Vector是用来解决列表(list)不能高效的随机访问的一种结构。Vector结构能够在“更高效”的固定时间内访问到列表中的任意元素。虽然这个时间会比访问头结点或者访问某数组元素所需的时间长一些，但至少这个时间也是个常量。
- **Range（等差数列）：**是一个有序的等差整数数列。比如说`1, 2, 3`就是一个Range。在Scala中创建一个Range类，需要用到两个预定义的方法`to`和`by`。
- **Hash Trie：**Hash Try对于快速查找和函数式的高效添加和删除操作上取得了很好的平衡，是高效实现不可变集合和关联数组(maps)的标准方法。从表现形式上看，Hash Try和Vector比较相似，都是树结构，且每个节点包含32个元素或32个子树，差别只是用不同的hash code替换了指向各个节点的向量值。
- **Red-Black Tree：**红黑树是一种平衡二叉树，对红黑树的最长运行时间随树的节点数成对数增长。Scala隐含的提供了不可变集合和映射的红黑树实现，您可以在TreeSet和TreeMap下使用这些方法。
- **BitSet（位集合）：**代表一个由小整数构成的容器，这些小整数的值表示了一个大整数被置1的各个位。比如说，一个包含3、2和0的bit集合可以用来表示二进制数1101和十进制数13。
- **ListMap：**表示一个保存`key -> val`的链表。一般情况下，ListMap操作都需要遍历整个列表，所以操作的运行时间也同列表长度成线性关系。在Scala中很少使用，因为标准的不可变映射通常速度会更快。唯一的例外是，在构造映射时由于某种原因，链表中靠前的元素被访问的频率大大高于其他的元素。

看过了Scala的不可变容器类，这些是标准库中最常用的。现在来看一下可变容器类：

- **ArrayBuffer：**对数组缓冲的大多数操作，其速度与数组本身无异。因为这些操作直接访问、修改底层数组。另外，数组缓冲可以进行高效的尾插数据。追加操作均摊下来只需常量时间。因此，数组缓冲可以高效的建立一个有大量数据的容器，无论是否总有数据追加到尾部。
- **ListBuffer：**类似于数组缓冲。区别在于内部实现是链表，而非数组。如果你想把构造完的缓冲转换为列表，那就用列表缓冲，别用数组缓冲。
- **StringBuilder：**用来构造字符串。作为常用的类，字符串构造器已导入到默认的命名空间。直接用`new StringBuilder`就可创建字符串构造器。
- **LinkedList：**链表是可变序列，它由一个个使用next指针进行链接的节点构成，是最佳的顺序遍历序列。此外，链表可以很容易去插入一个元素或链接到另一个链表。`DoubleLinkedList`双向链表。
- **ArraySeq：**是具有固定大小的可变序列。在它的内部，用一个`Array[Object]`来存储元素。如果你想拥有Array的性能特点，又想建立一个泛型序列实例，但是你又不知道其元素的类型，在运行阶段也无法提供一个ClassManifest，那么你通常可以使用ArraySeq。
- **HashTable：**能够根据一种良好的hash codes分配机制来存放对象，它的速度会非常快。Scala中默认的可变map和set都是基于HashTable的。你也可以直接用`mutable.HashSet`和`mutable.HashMap`来访问它们。

## 与Java共舞
某些时候，你需要将一种容器类型转换成另外一种类型。例如，你可能想要像访问Scala容器一样访问某个Java容器，或者你可能想将一个Scala容器像Java容器一样传递给某个Java方法。在Scala中，这是很容易的，因为Scala提供了大量的方法来隐式转换所有主要的Java和Scala容器类型。如下：
```
Iterator <=> java.util.Iterator
Iterator <=> java.util.Enumeration
Iterable <=> java.lang.Iterable
Iterable <=> java.util.Collection
mutable.Buffer <=> java.util.List
mutable.Set <=> java.util.Set
mutable.Map <=> java.util.Map
mutable.ConcurrentMap <=> java.util.concurrent.ConcurrentMap
```

使用这些转换很简单，只需从JavaConversions对象中`import`它们即可。就可以在Scala容器和与之对应的Java容器之间进行隐式转换了：
```
scala> import collection.JavaConversions._
scala> import collection.mutable._
scala> val jul: java.util.List[Int] = ArrayBuffer(1, 2, 3)
jul: java.util.List[Int] = [1, 2, 3]
scala> val buf: Seq[Int] = jul
buf: scala.collection.mutable.Seq[Int] = ArrayBuffer(1, 2, 3)
scala> val m: java.util.Map[String, Int] = HashMap("abc" -> 1, "hello" -> 2)
m: java.util.Map[String, Int] = {hello=2, abc=1}
```

可以通过`JavaConverters package`轻松地在Java和Scala的集合类型之间转换。它用`asScala`装饰常用的Java集合以和`asJava`方法装饰Scala集合：
```
scala> import scala.collection.JavaConverters._
import scala.collection.JavaConverters._
scala> val s1 = new scala.collection.mutable.ListBuffer[Int]
s1: scala.collection.mutable.ListBuffer[Int] = ListBuffer()
scala> val j1 = s1.asJava
j1: java.util.List[Int] = []
scala> val s2 = j1.asScala
s2: scala.collection.mutable.Buffer[Int] = ListBuffer()
scala> assert(s1 eq s2)
scala> val l1 = List(1, 2, 3)
l1: List[Int] = List(1, 2, 3)
scala> l1.asJava
res2: java.util.List[Int] = [1, 2, 3]
scala> l1.asJava.asScala
res3: scala.collection.mutable.Buffer[Int] = Buffer(1, 2, 3)
```

>在Scala内部，这些转换是通过一系列“包装”对象完成的，这些对象会将相应的方法调用转发至底层的容器对象。所以容器不会在Java和Scala之间拷贝来拷贝去。一个值得注意的特性是，如果你将一个Java容器转换成其对应的Scala容器，然后再将其转换回同样的Java容器，最终得到的是一个和一开始完全相同的容器对象（译注：这里的相同意味着这两个对象实际上是指向同一片内存区域的引用，容器转换过程中没有任何的拷贝发生）。

还有一些Scala容器类型可以转换成对应的Java类型，但是并没有将相应的Java类型转换成Scala类型的能力，它们是：
```
Seq => java.util.List
mutable.Seq => java.util.List
Set => java.util.Set
Map => java.util.Map
```

因为Java并未区分可变容器不可变容器类型，所以，虽然能将`scala.immutable.List`转换成`java.util.List`，但所有的修改操作都会抛出`UnsupportedOperationException`。参见下例：
```
scala> val jul = List(1, 2, 3).asJava
jul: java.util.List[Int] = [1, 2, 3]
scala> jul.add(7)
java.lang.UnsupportedOperationException
        at java.util.AbstractList.add(AbstractList.java:131)
```

## 函数
定义函数，返回值类型可省略，`=`后面可以是块或者表达式。无参数的函数调用时可以省略括号。
```
def f(x: Int) = { x*x }
def f(x: Any): Unit = println(x)
reply()
reply
```

一个参数时可以使用`infix`写法，如下：
```
val x = 3
x.+(3)
x + 3
```

匿名函数(lambda表达式)，如下：
```
(argument) => // funtion body
```

可以使用`_`部分应用一个函数，结果将得到另一个函数：
```
scala> def adder(m: Int, n: Int) = m + n
adder: (m: Int, n: Int)Int
scala> val add2 = adder(2, _:Int)
add2: Int => Int = <function1>
scala> add2(3)
res14: Int = 5
```

柯里化函数。有时会有这样的需求：允许别人一会在你的函数上应用一些参数，然后又应用另外的一些参数。例如一个乘法函数，在一个场景需要选择乘数，而另一个场景需要选择被乘数：
```
scala> def division(m: Int)(n: Int): Int = m / n
division: (m: Int)(n: Int)Int
scala> division(6)(2)
res18: Int = 3
scala> val divTwo = division(6) _
divTwo: Int => Int = <function1>
scala> divTwo(2)
res19: Int = 3
```

## 样例类
使用样例类可以方便得存储和匹配类的内容。你不用`new`关键字就可以创建它们：
```
scala> case class Calculator(brand: String, model: String)
defined class Calculator
scala> val hp20b = Calculator("HP", "20b")
hp20b: Calculator = Calculator(HP,20b)
```

样例类基于构造函数的参数，自动地实现了`apply, unapply, equal, copy, toString`：
```
scala> val hp20B = Calculator("HP", "20b")
hp20B: Calculator = Calculator(HP,20b)
scala> hp20b == hp20B
res20: Boolean = true
```

样例类就是被设计用在模式匹配中的。一个例子：
```
val hp20b = Calculator("HP", "20B")
val hp30b = Calculator("HP", "30B")
def calcType(calc: Calculator) = calc match {
    case Calculator("HP", "20B") => "financial"
    case Calculator("HP", "30B") => "business"
    case Calculator("HP", "48G") => "scientific"
    case Calculator(ourBrand, ourModel) => "Calculator: %s %s is of unknown type".format(ourBrand, ourModel)
}
```

最后一句也可以这样写`case Calculator(_, _) => "Calculator of unknown type"`，或者不指定匹配类型`case _ => "Calculator of unknown type"`，或者将匹配的值重新命名`case c@Calculator(_, _) => "Calculator: %s of unknown type".format(c)`。

## 模式匹配
这是Scala中最有用的部分之一。如下：
```
val times = 1
times match {
    case 1 => "one"
    case 2 => "two"
    case _ => "some other number"
}
```

你可以使用`match`来分别处理不同类型的值：
```
o match {
    case i: Int if i < 0 => i - 1
    case i: Int => i + 1
    case d: Double if d < 0.0 => d - 0.1
    case d: Double => d + 0.1
    case text: String => text + "s"
}
```

元组可以很好得与模式匹配相结合：
```
val hostPort = ("localhost", 80)
hostPort match {
    case ("localhost", port) => println("default", port)
    case (host, port) => println(host, port)
}
```

## _*
`_*`可以用作模式匹配任意数量的参数：
```
case class A(n: Int*)
val a = A(1, 2, 3, 4, 5)
a match {
    case A(1, 2, _*) =>
}
```

你可以将可变参数绑定到一个值上：
```
a match {
    case A(1, 2, as@_*) => println(as)
}
```

另外`_*`还可以作为辅助类型描述。如将集合作为可变参数传递：
```
val l = 1 :: 2 :: 3 :: Nil
val a = A(l:_*)
```

## Option
`Option`是一个表示有可能包含值的容器。`Option`本身是泛型的，并且有两个子类：`Some[T]`或`None`。`Map.get`使用`Option`作为其返回值，表示这个方法也许不会返回你请求的值：
```
scala> val numbers = Map("one" -> 1, "two" -> 2)
numbers: scala.collection.immutable.Map[String,Int] = Map(one -> 1, two -> 2)
scala> numbers.get("two")
res34: Option[Int] = Some(2)
scala> numbers.get("three")
res35: Option[Int] = None
```

现在我们的数据似乎陷在`Option`中了，我们怎样获取这个数据呢？
```
scala> if(res34.isDefined) res34.get else 0  // 方法0
res0: Int = 2
scala> if(res35.isDefined) res35.get else 0
res1: Int = null
scala> res34.getOrElse(0)  // 方法1
res42: Int = 2
scala> res35.getOrElse(0)
res43: Int = 0
scala> res34 match {  // 方法2
    case Some(n) => n * 2
    case None => 0
}
res44: Int = 4
```

## apply,unapply,update,unapplySeq
当类或对象有一个主要用途的时候，`apply`方法为你提供了一个很好的语法糖。比如`a`是一个对象，`a.apply(x)`则可以简化为`a(x)`。`unapply`方法是抽取器(Extractor)，经常用在模式匹配上，例如：
```
val foo = Foo(1)  // 调用apply
foo match {
    case Foo(x) => println(x)  // 调用unapply
}
class Foo(val x: Int) {}
object Foo {
    def apply(x: Int) = new Foo(x)
    def unapply(f: Foo) = Some(f.x)
}
```

与`apply`方法类似，`update`也是一个魔法方法。对于一个实例`a`, Scala可以将`a.update(x,y)`简化为`a(x)=y`：
```
val bar = Bar(10)
bar(0) = 1  // 调用update
println(bar(0))
class Bar(n: Int) {
    val a = Array[Int](n)
    def apply(n: Int) = a(n)
    def update(n: Int, v: Int) = a(n) = v
}
object Bar {
    def apply(n: Int) = new Bar(n)
}
```

类似`unapply`，`unapplySeq`用来从对象中抽取序列，常用在模式匹配上：
```
val m = M(1)
m match {
    case M(n1,others@_*) => println(others)
}
class M {}
object M {
    def apply(a: Int*): M = new M
    def unapplySeq(m: M): Option[Seq[Int]] = Some(1 :: 2 :: 3 :: Nil)
}
```

## 函数组合子
`List(1, 2, 3) map squared`对列表中的每一个元素都应用了`squared`函数，并返回一个新的列表`List(1, 4, 9)`。我们称这个操作为`map`组合子。它们常被用在标准的数据结构上。

`map`对列表中的每个元素应用一个函数，返回应用后的元素所组成的列表：
```
scala> List(1, 2, 3).map((i: Int) => i * 2)
res51: List[Int] = List(2, 4, 6)
scala> def times(i: Int): Int = i * 2
times: (i: Int)Int
scala> List(1, 2, 3).map(times)
res52: List[Int] = List(2, 4, 6)
scala> List(1, 2, 3).map(times(_))
res53: List[Int] = List(2, 4, 6)
```

`foreach`很像`map`，但没有返回值。`foreach`仅用于有副作用的函数：
```
scala> List(1, 2, 3).foreach((i: Int) => i * 2)
scala> List(1, 2, 3).foreach((i: Int) => i * 2).getClass
res64: Class[Unit] = void
```

`filter`移除任何对传入函数计算结果为`false`的元素：
```
scala> List(1, 2, 3, 4).filter((i: Int) => i % 2 == 0)
res70: List[Int] = List(2, 4)
```

`zip`将两个列表的内容聚合到一个对偶列表中：
```
scala> List(1, 2, 3).zip(List("a", "b", "c"))
res71: List[(Int, String)] = List((1,a), (2,b), (3,c))
```

>返回的列表长度取决于较短的列表，只要有一个列表到达了末尾`zip`函数就停止了。可以使用`zipAll`函数来对较长列表的剩余元素进行处理。

`partition`将使用给定的谓词函数分割列表：
```
scala> val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
numbers: List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
scala> numbers.partition(_ % 2 == 0)
res72: (List[Int], List[Int]) = (List(2, 4, 6, 8, 10),List(1, 3, 5, 7, 9))
```

`find`返回集合中第一个匹配谓词函数的元素：
```
scala> numbers.find((i: Int) => i > 5)
res73: Option[Int] = Some(6)
```

`drop`将删除前i个元素，`dropWhile`将删除元素直到第一个不匹配：
```
scala> numbers.drop(5)
res76: List[Int] = List(6, 7, 8, 9, 10)
scala> numbers.dropWhile(_ % 2 != 0)
res77: List[Int] = List(2, 3, 4, 5, 6, 7, 8, 9, 10)
```

`fold`将一种格式的输入转化成另外一种格式输出。`0`为初始值，`m`为累加器：
```
scala> List(1, 2, 3).fold(0){(m: Int, n: Int) => println("m: " + m + " n: " + n); m + n}
m: 0 n: 1
m: 1 n: 2
m: 3 n: 3
res85: Int = 6
```

`reduce`和`fold`很像，`reduce`返回值的类型必须和列表的元素类型相关，类型本身或父类。`fold`没有这种限制，但`fold`必须给定一个初始值，可以说`reduce`是`fold`的一种特殊情况：
```
scala> val sum = (x: Int, y: Int) => {println(x, y); x + y}
sum: (Int, Int) => Int = <function2>
scala> (1 to 5).reduce(sum)
(1,2)
(3,3)
(6,4)
(10,5)
res99: Int = 15
```

`flatten`将嵌套结构扁平化为一个层次的集合：
```
scala> List(List(1, 2), List(3, 4)).flatten
res87: List[Int] = List(1, 2, 3, 4)
```

`flatMap`是一种常用的组合子，结合映射和扁平化。`flatMap`需要一个处理嵌套列表的函数，然后将结果串连起来。可以看做是“先映射后扁平化”的快捷操作：
```
scala> List(List(1, 2), List(3, 4)).flatMap(x => x.map(_ * 2))
res88: List[Int] = List(2, 4, 6, 8)
scala> List(List(1, 2), List(3, 4)).map(x => x.map(_ * 2)).flatten
res89: List[Int] = List(2, 4, 6, 8)
```

扩展函数组合子。现在我们已经学过集合上的一些函数。我们将尝试写自己的函数组合子。有趣的是，上面所展示的每一个函数组合子都可以用`fold`方法实现。让我们看一个例子：
```
scala> def ourMap(numbers: List[Int], fn: Int => Int): List[Int] = {
    numbers.foldRight(List[Int]()){(x: Int, xs: List[Int]) => fn(x) :: xs}
}
ourMap: (numbers: List[Int], fn: Int => Int)List[Int]
scala> ourMap(numbers, _ * 2)
res94: List[Int] = List(2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
```

## 参考资料：
- [Scala Documentation](http://docs.scala-lang.org/zh-cn/overviews/)
- [Scala 课堂!](http://twitter.github.io/scala_school/zh_cn/index.html)
- [Scala简明教程](http://colobu.com/2015/01/14/Scala-Quick-Start-for-Java-Programmers/)