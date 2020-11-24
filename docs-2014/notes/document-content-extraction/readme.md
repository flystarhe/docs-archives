title: 常见格式文件内容抽取（Tika）
date: 2016-08-25
tags: [Tika,内容抽取]
---
面对各种类型的文件，大部分应用都希望有一个简单的基于文本的表示可供内部使用。这样就可以允许使用Scala、Python和其他编程语言中的字符串处理库。

<!--more-->
## Apache Tika
Tika是一个允许从许多不同源抽取内容的框架，这些源包括Word、PDF、文本及一些其他类型。除了对不同的抽取库进行的包装之外，Tika也提供了MIME检测功能，这样Tika就能自动检测内容的类型从而调用正确的库进行处理。

Tika的工作机理与SAX处理XML的机理很像。Tika从底层的内容格式（PDF、Word等）中抽取信息，然后提供回调事件，该事件可以被应用所处理。

## Usage Tika
与Tika交互就像实例化一个`Tika Parser`类那么简单，该类提供了单个方法：
```java
void parse(InputStream stream, ContentHandler handler,
    Metadata metadata, ParseContext parseContext)
    throws IOException SAXException, TikaException;
```

使用`parse`方法，你只需要将内容作为一个`InputStream`传入，而内容事件将被`ContentHandler`的应用实现所处理。内容中的元数据会跳到`Metadata`实例中，其核心是一张哈希表。

在`Parser`这个层次，Tika带了多个可用的实现，其中一个包含了Tika所支持的每个具体MIME类型，而另一个`AutoDetectParser`能够自动识别内容的MIME类型。Tika还带了多个常见抽取场景下的`ContentHandler`实现，这些场景包括比如只抽取文件的正文部分等等。

>build.sbt: libraryDependencies += "org.apache.tika" % "tika-parsers" % "1.13"

## Case of HTML
假设你的HTML文件如下：
```html
<html>
<head>
<title>Best Pizza Joints in America</title>
</head>
<body>
<p>The best pizza in the US is <a href="http://antoniospizzas.com">Antonio's Pizza</a>.</p>
<p>It is located in Amherst. MA.</p>
</body>
</html>
```

你很可能想从中抽取标题、正文及可能的链接。Tika使这一切都很容易：
```scala
import java.io.FileInputStream
import org.apache.tika.metadata.Metadata
import org.apache.tika.parser.ParseContext
import org.apache.tika.parser.html.HtmlParser
import org.apache.tika.sax.{TeeContentHandler, LinkContentHandler, BodyContentHandler}

object TestTika {
  def main(args: Array[String]) {
    val input = new FileInputStream("doc1.html")
    val text = new BodyContentHandler() //正文的ContentHandler
    val links = new LinkContentHandler() //链接的ContentHandler
    val handler = new TeeContentHandler(text, links) //合并多个ContentHandler
    val metadata = new Metadata() //用于保存元数据
    val parser = new HtmlParser() //Html解析器
    val context = new ParseContext()
    parser.parse(input, handler, metadata, context) //执行解析过程
    println("title", metadata.get("title"))
    println("body", text.toString)
    println("links", links.getLinks)
  }
}
```

运行结果如下：
```
(title,Best Pizza Joints in America)
(body,
The best pizza in the US is Antonio's Pizza.
It is located in Amherst. MA.
)
(links,[<a href="http://antoniospizzas.com">Antonio's Pizza</a>])
```

## Case of PDF
上述HTML的例子中，使用`HtmlParser`来解析内容，但是大部分情况下你可能想使用Tika内置的`AutoDetectParser`类来解析文件。解析PDF实例：
```scala
import java.io.FileInputStream
import org.apache.tika.metadata.Metadata
import org.apache.tika.parser.{AutoDetectParser, ParseContext}
import org.apache.tika.parser.html.HtmlParser
import org.apache.tika.sax.{TeeContentHandler, LinkContentHandler, BodyContentHandler}

object TestTika {
  def main(args: Array[String]) {
    val input = new FileInputStream("doc2.pdf")
    val textHandler = new BodyContentHandler() //正文的ContentHandler
    val metadata = new Metadata() //用于保存元数据
    val parser = new AutoDetectParser() //自动估计文档MIME类型
    val context = new ParseContext()
    parser.parse(input, textHandler, metadata, context) //执行解析过程
    println("metadata", metadata.names().toList)
    println("body", textHandler.toString)
  }
}
```

运行结果如下：
```
(metadata,List(pdf:PDFVersion, X-Parsed-By, creator, xmp:CreatorTool, access_permission:modify_annotations, meta:author, access_permission:can_print_degraded, meta:creation-date, created, dc:creator, access_permission:extract_for_accessibility, access_permission:assemble_document, xmpTPg:NPages, Creation-Date, dcterms:created, dc:format, access_permission:extract_content, access_permission:can_print, access_permission:fill_in_form, pdf:encrypted, Author, producer, access_permission:can_modify, Content-Type))
(body,...
```

## Summary
Tika能够很容易处理所有不同文件格式这个事实确实是一个好消息，更好的一个消息是：Tika已经通过一个称为`Solr Content Extraction`库集成到ApacheSolr中。