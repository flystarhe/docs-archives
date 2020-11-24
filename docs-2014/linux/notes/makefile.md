title: Linux Makefile学习笔记
date: 2016-10-12
tags: [Linux,Makefile]
---
make命令执行时，需要一个Makefile文件，以告诉make命令需要怎么样做。Makefile带来的好处就是——自动化编译，一旦写好，只需要一个make命令，整个工程完全自动编译，极大的提高了软件开发的效率。这里针对GNU的make进行讲述，环境是CentOS。

<!--more-->
## Makefile的规则
在讲述这个Makefile之前，还是让我们先来粗略地看一看Makefile的规则。
```
target ... : prerequisites ...
command
...
```

`target`也就是一个目标文件，可以是Object File，也可以是执行文件。还可以是一个标签（Label），对于标签这种特性，在后续的“伪目标”章节中会有叙述。`prerequisites`就是，要生成那个`target`所需要的文件或是目标。`command`也就是需要执行的命令。（任意的Shell命令）

这是一个文件的依赖关系，也就是说`target`依赖于`prerequisites`，其生成规则定义在`command`中。就是说，`prerequisites`中如果有一个以上的文件比`target`文件要新的话，`command`所定义的命令就会被执行。这就是Makefile的规则。也就是Makefile中最核心的内容。

## 一个简单的示例
如果一个工程有3个头文件，和8个C文件，我们为了完成下面所述的三个规则：

1. 如果这个工程没有编译过，那么我们的所有C文件都要编译并被链接。
2. 如果这个工程的某几个C文件被修改，那么我们只编译被修改的C文件并链接目标程序。
3. 如果这个工程的头文件被改变了，那么我们需要编译引用了这几个头文件的C文件并链接目标程序。

我们的Makefile应该是下面的这个样子的：
```
edit : main.o kbd.o command.o display.o /
insert.o search.o files.o utils.o
    cc -o edit main.o kbd.o command.o display.o insert.o search.o files.o utils.o

main.o : main.c defs.h
    cc -c main.c
kbd.o : kbd.c defs.h command.h
    cc -c kbd.c
command.o : command.c defs.h command.h
    cc -c command.c
display.o : display.c defs.h buffer.h
    cc -c display.c
insert.o : insert.c defs.h buffer.h
    cc -c insert.c
search.o : search.c defs.h buffer.h
    cc -c search.c
files.o : files.c defs.h buffer.h command.h
    cc -c files.c
utils.o : utils.c defs.h
    cc -c utils.c

clean :
    rm edit main.o kbd.o command.o display.o insert.o search.o files.o utils.o
```

反斜杠（/）是换行符的意思。这样比较便于Makefile的易读。我们可以把这个内容保存在文件为“Makefile”或“makefile”的文件中，然后在该目录下直接输入命令“make”就可以生成执行文件edit。如果要删除执行文件和所有的中间目标文件，那么，只要简单地执行一下“make clean”就可以了。

在这个makefile中，目标文件`target`包含：执行文件`edit`和中间目标文件`*.o`，依赖文件`prerequisites`就是冒号后面的那些`.c`和`.h`文件。每一个`.o`文件都有一组依赖文件，而这些`.o`文件又是执行文件`edit`的依赖文件。依赖关系的实质上就是说明了目标文件是由哪些文件生成的，换言之，目标文件是哪些文件更新的。

在定义好依赖关系后，后续的那一行定义了如何生成目标文件的操作系统命令，一定要以一个Tab键作为开头。记住，make并不管命令是怎么工作的，他只管执行所定义的命令。make会比较targets文件和prerequisites文件的修改日期，如果prerequisites文件的日期要比targets文件的日期要新或target不存在的话，make就会执行后续定义的命令。

这里要说明一点的是，clean不是一个文件，它只不过是一个动作名字，有点像C语言中的lable一样，其冒号后什么也没有，make就不会自动去找文件的依赖性，也就不会自动执行其后所定义的命令。要执行其后的命令，就要在make命令后明显得指出这个lable的名字。这样的方法非常有用，我们可以在一个makefile中定义不用的编译或是和编译无关的命令，比如程序的打包、备份等。

## Makefile中使用变量
在上面的例子中，edit的规则`main.o kbd.o ..`字符串被重复了两次，如果我们的工程需要加入一个新的`.o`文件，那么我们需要在两个地方编辑（应该是三个地方，还有一个地方在clean中）。如果Makefile变得复杂，那么我们就有可能会忘掉一些需要加入的地方，而导致编译失败。所以，为了Makefile的易维护，在Makefile中我们可以使用变量。

比如，我们声明一个变量，在Makefile一开始就这样定义：
```
objects = main.o kbd.o command.o display.o insert.o search.o files.o utils.o
```

于是，我们就可以很方便地在我们的Makefile中以`$(objects)`的方式来使用这个变量了，于是我们的改良版Makefile就变成下面这个样子：
```
objects = main.o kbd.o command.o display.o insert.o search.o files.o utils.o

edit : $(objects)
    cc -o edit $(objects)

main.o : main.c defs.h
    cc -c main.c
kbd.o : kbd.c defs.h command.h
    cc -c kbd.c
command.o : command.c defs.h command.h
    cc -c command.c
display.o : display.c defs.h buffer.h
    cc -c display.c
insert.o : insert.c defs.h buffer.h
    cc -c insert.c
search.o : search.c defs.h buffer.h
    cc -c search.c
files.o : files.c defs.h buffer.h command.h
    cc -c files.c
utils.o : utils.c defs.h
    cc -c utils.c

clean :
    rm edit $(objects)
```

于是如果有新的`.o`文件加入，我们只需简单地修改一下`objects`变量就可以了。

## 让make自动推导
GNU的make很强大，它可以自动推导文件以及文件依赖关系后面的命令，我们没必要去在每一个`.o`文件后都写类似的命令，make会自动识别并自己推导命令。只要make看到一个`.o`文件，它就会自动的把`.c`文件加在依赖关系中，如果make找到一个`whatever.o`，那么`whatever.c`，就会是`whatever.o`的依赖文件。并且`cc -c whatever.c`也会被推导出来。于是，我们的Makefile再也不用写得这么复杂。我们的是新的Makefile又出炉了：
```
objects = main.o kbd.o command.o display.o insert.o search.o files.o utils.o

edit : $(objects)
    cc -o edit $(objects)

main.o : defs.h
kbd.o : defs.h command.h
command.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o : defs.h buffer.h command.h
utils.o : defs.h

.PHONY : clean
clean :
    -rm edit $(objects)
```

这种方法，也就是make的“隐晦规则”。上面文件内容中：`.PHONY`表示`clean`是个伪目标文件；在rm命令前面加了一个减号的意思是，也许某些文件出现问题，但不要管，继续做后面的事。

## make文件搜寻
在一些大的工程中，有大量的源文件，我们通常的做法是把这许多的源文件分类，并存放在不同的目录中。所以，当make需要去找寻文件的依赖关系时，你可以在文件前加上路径，但最好的方法是把一个路径告诉make，让make在自动去找。

Makefile文件中的特殊变量`VPATH”`是完成这个功能的，如果没有指明这个变量，make只会在当前的目录中去找寻依赖文件和目标文件。如果定义了这个变量，make就会在当当前目录找不到的情况下，到所指定的目录中去找寻文件了。
```
VPATH = src:../headers
```

上面的的定义指定两个目录，`src`和`../headers`，make会按照这个顺序进行搜索。目录由“冒号”分隔。（当然，当前目录永远是最高优先搜索的地方）

另一个设置文件搜索路径的方法是使用make的`vpath`关键字（注意，它是全小写的），这不是变量，这是一个make的关键字，这和上面提到的那个VPATH变量很类似，但是它更为灵活。它可以指定不同的文件在不同的搜索目录中。这是一个很灵活的功能。它的使用方法有三种：

1. `vpath <pattern> <directories>`:为符合模式`<pattern>`的文件指定搜索目录`<directories>`。
2. `vpath <pattern>`:清除符合模式`<pattern>`的文件的搜索目录。
3. `vpath`:清除所有已被设置好了的文件搜索目录。

`vapth`使用方法中的`<pattern>`需要包含`%`字符。`%`的意思是匹配零或若干字符，例如，`%.h`表示所有以`.h`结尾的文件。

## Makefile多目标与自动化变量
Makefile的规则中的目标可以不止一个，其支持多目标。有可能我们的多个目标同时依赖于一个文件，并且其生成的命令大体类似。于是我们就能把其合并起来。当然，多个目标的生成规则的执行命令是同一个，这可能会可我们带来麻烦，不过好在我们的可以使用一个自动化变量`$@`（关于自动化变量，将在后面讲述），这个变量表示着目前规则中所有的目标的集合，这样说可能很抽象，还是看一个例子吧。
```
bigoutput littleoutput : text.g
generate text.g -$(subst output,,$@) > $@
```

上述规则等价于：
```
bigoutput : text.g
generate text.g -big > bigoutput
littleoutput : text.g
generate text.g -little > littleoutput
```

其中，`-$(subst output,,$@)`中的`$`表示执行一个Makefile的函数，函数名为`subst`，后面的为参数。关于函数，将在后面讲述。这里的这个函数是截取字符串的意思，`$@`表示目标的集合，就像一个数组，`$@`依次取出目标，并执于命令。

## Makefile的命令书写
每条规则中的命令和操作系统Shell的命令行是一致的。make会一按顺序一条一条的执行命令，每条命令的开头必须以`Tab`键开头，除非命令是紧跟在依赖规则后面的分号后的。在命令行之间中的空格或是空行会被忽略，但是如果该空格或空行是以Tab键开头的，那么make会认为其是一个空命令。

通常，make会把其要执行的命令行在命令执行前输出到屏幕上。当我们用`@`字符在命令行前，那么这个命令将不被make显示出来，最具代表性的例子是，我们用这个功能来像屏幕显示一些信息。如：
```
@echo 正在编译XXX模块..
```

当make执行时，会输出“正在编译XXX模块..”字串，但不会输出命令，如果没有`@`，那么make将输出：
```
echo 正在编译XXX模块..
正在编译XXX模块..
```

需要注意的是，如果你要让上一条命令的结果应用在下一条命令时，你应该使用分号分隔这两条命令。比如你的第一条命令是`cd`，你希望第二条命令得在`cd`之后的基础上运行，那么你就不能把这两条命令写在两行上，而应该把这两条命令写在一行上，用分号分隔。如：
```
示例一：
exec:
cd /home/flylab
pwd

示例二：
exec:
cd /home/flylab; pwd
```

当我们执行`make exec`时，第一个例子中的`cd`没有作用，`pwd`会打印出当前的Makefile目录，而第二个例子中，`cd`就起作用了，`pwd`会打印出`/home/flylab`。

## 自动化变量
在模式规则中，目标和依赖文件都是一系例的文件，那么我们如何书写一个命令来完成从不同的依赖文件生成相应的目标？因为在每一次的对模式规则的解析时，都会是不同的目标和依赖文件。

自动化变量就是完成这个功能的。在前面我们已经对自动化变量有所提涉，相信你看到这里已对它有一个感性认识了。所谓自动化变量，就是这种变量会把模式中所定义的一系列的文件自动地挨个取出，直至所有的符合模式的文件都取完了。这种自动化变量只应出现在规则的命令中。

下面是所有的自动化变量及其说明：

    $@

表示规则中的目标文件集。在模式规则中，如果有多个目标，那么`$@`就是匹配于目标中模式定义的集合。

    $%

仅当目标是函数库文件中，表示规则中的目标成员名。例如，如果一个目标是`foo.a(bar.o)`，那么`$%`就是`bar.o`，`$@`就是`foo.a`。如果目标不是函数库文件（Unix下是`.a`，Windows下是`.lib`），那么其值为空。

    $<

依赖目标中的第一个目标名字。如果依赖目标是以模式（即`%`）定义的，那么`$<`将是符合模式的一系列的文件集。注意，其是一个一个取出来的。

    $?

所有比目标新的依赖目标的集合，以空格分隔。

    $^

所有的依赖目标的集合，以空格分隔。如果在依赖目标中有多个重复的，这个变量会去除重复的依赖目标，只保留一份。

    $+

这个变量很像`$^`，也是所有依赖目标的集合。只是它不去除重复的依赖目标。

    $*

这个变量表示目标模式中`%`及其之前的部分。如果目标是`dir/a.foo.b`，并且目标的模式是`a.%.b`，那么`$*`的值就是`dir/a.foo`。这个变量对于构造有关联的文件名是比较有较。如果目标中没有模式的定义，那么`$*`也就不能被推导出，但是，如果目标文件的后缀是make所识别的，那么`$*`就是除了后缀的那一部分。例如：如果目标是`foo.c`，因为`.c`是make所能识别的后缀名，所以`$*`的值就是`foo`。这个特性是GNU make的，很有可能不兼容于其它版本的make，所以你应该尽量避免使用`$*`，除非是在隐含规则或是静态模式中。如果目标中的后缀是make所不能识别的，那么`$*`就是空值。

在上述所列出来的自动量变量中。四个变量`$@ $< $% $*`在扩展时只会有一个文件，而另三个的值是一个文件列表。

## 参考资料：
- [Linux Makefile详细语法](http://blog.163.com/bical@126/blog/static/479354942013411114118416/)