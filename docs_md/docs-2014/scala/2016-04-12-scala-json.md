title: Gson使用笔记
date: 2016-04-12
tags: [Scala,JSON,Gson]
---
Gson是Google开发的Java API，用于转换Java对象和Json对象，本文讨论并提供了使用API的简单实例，更多关于Gson的资料可以访问[url](http://sites.google.com/site/gson/)。

<!--more-->
## Array and Json

    val j_array = Array("one", "two", "three")
    /**
      * 数组 to Json
      */
    val s_array = m_gson.toJson(j_array)
    println(s_array)
    /**
      * Json to 数组
      */
    val j_array_new = m_gson.fromJson(s_array, classOf[Array[String]])
    for (i <- j_array_new) println(i)

## List and Json

    val j_list = new util.LinkedList[Person]()
    j_list.add(new Person("a", 1))
    j_list.add(new Person("b", 2))
    j_list.add(new Person("c", 3))
    /**
      * 列表 to Json
      */
    val s_list = m_gson.toJson(j_list)
    println(s_list)
    /**
      * Json to 列表
      */
    val j_list_new = m_gson.fromJson(s_list, classOf[util.List[Person]])
    for (i <- j_list_new.toArray()) println(i)

Java对象声明：

    class Person(inam: String, iage: Int) {
      var name: String = inam
      var age: Int = iage
    }

## Map and Json

    val j_map = new util.HashMap[String, String]()
    j_map.put("k1", "v1")
    j_map.put("k2", "v2")
    j_map.put("k3", "v3")
    /**
      * Map to Json
      */
    val s_map = m_gson.toJson(j_map)
    println(s_map)
    /**
      * Json to Map
      */
    val j_map_new = m_gson.fromJson(s_map, classOf[util.Map[String, String]])
    for (i <- j_map_new.keySet().toArray()) println(i + "/" + j_map_new.get(i))

## 序列化

    public class Book {
      private String[] authors;
      private String isbn10;
      private String isbn13;
      private String title;
    }

考虑上面的Java对象，假如我们需要将其序列化为下面这个Json对象。

    {
        "title": "Java Puzzlers: Traps, Pitfalls, and Corner Cases",
        "isbn-10": "032133678X",
        "isbn-13": "978-0321336781",
        "authors": ["Joshua Bloch", "Neal Gafter"]
    }

Gson不需要任何特殊配置就可以序列化Book类。Gson使用Java字段名称作为Json字段的名称，并赋予对应的值。如果仔细地看一下上面的那个Json示例会发现，ISBN字段包含一个减号：isbn-10和isbn-13。不幸的是，使用默认配置不能将这些字段包含进来。解决问题的办法之一就是使用注解[Gson注解示例](http://www.javacreed.com/gson-annotations-example/)。使用注解可以自定义Json字段的名称，Gson将会以注解为准进行序列化。另一个方法就是使用[JsonSerialiser](http://www.javacreed.com/gson-serialiser-example/)，如下所示：

    public class BookSerialiser implements JsonSerializer {
        @Override
        public JsonElement serialize(final Book book, final Type typeOfSrc, final JsonSerializationContext context) {
            final JsonObject jsonObject = new JsonObject();
            jsonObject.addProperty("title", book.getTitle());
            jsonObject.addProperty("isbn-10", book.getIsbn10());
            jsonObject.addProperty("isbn-13", book.getIsbn13());
            final JsonArray jsonAuthorsArray = new JsonArray();
            for (final String author : book.getAuthors()) {
                final JsonPrimitive jsonAuthor = new JsonPrimitive(author);
                jsonAuthorsArray.add(jsonAuthor);
            }
            jsonObject.add("authors", jsonAuthorsArray);
            return jsonObject;
        }
    }

JsonSerializer接口要求类型是将要进行序列化的对象类型。在这个例子中，我们要序列化的Java对象是Book。`serialize()`方法的返回类型必须是一个JsonElement类型的实例。下面列出了JsonElement四种具体实现类型：

    JsonPrimitive: 例如一个字符串或整型
    JsonObject: 以JsonElement名字为索引的集合，Map<String,JsonElement>
    JsonArray: JsonElement的集合，元素可以是四种类型中的一种或者混合
    JsonNull: 值为null

在调用该序列化方法之前，我们需要将其注册到Gson中：

    // Configure Gson
    final GsonBuilder gsonBuilder = new GsonBuilder();
    gsonBuilder.registerTypeAdapter(Book.class, new BookSerialiser());
    gsonBuilder.setPrettyPrinting();
    final Gson gson = gsonBuilder.create();
    // Create Object
    final Book javaPuzzlers = new Book();
    javaPuzzlers.setTitle("Java Puzzlers: Traps, Pitfalls, and Corner Cases");
    javaPuzzlers.setIsbn10("032133678X");
    javaPuzzlers.setIsbn13("978-0321336781");
    javaPuzzlers.setAuthors(new String[] {"Joshua Bloch", "Neal Gafter"});
    // Format to Json
    final String json = gson.toJson(javaPuzzlers);
    System.out.println(json);

接下来将会描述怎么序列化嵌套对象，所谓嵌套对象是指在其它对象内部的对象。在此我们将会引入一个新的实体`author`。形成了这样一个包含title和ISBN连同author列表的`book`。这个例子得到一个新实体的Json对象，与前面的Json对象不同，就像下面那样：

    {
        "title": "Java Puzzlers: Traps, Pitfalls, and Corner Cases",
        "isbn": "032133678X",
        "authors": [
            {
                "id": 1,
                "name": "Joshua Bloch"
            },
            {
                "id": 2,
                "name": "Neal Gafter"
            }
        ]
    }

Java对象声明：

    class Author {
        private int id;
        private String name;
    }
    class Book {
        private Author[] authors;
        private String isbn;
        private String title;
    }

author字段从String数组变成了Author数组。因此必须修改下BookSerialiser类来兼容这一改变，如下：

    public class BookSerialiser implements JsonSerializer {
        @Override
        public JsonElement serialize(final Book book, final Type typeOfSrc, final JsonSerializationContext context) {
            final JsonObject jsonObject = new JsonObject();
            jsonObject.addProperty("title", book.getTitle());
            jsonObject.addProperty("isbn", book.getIsbn());
            final JsonElement jsonAuthors = context.serialize(book.getAuthors());
            jsonObject.add("authors", jsonAuthors);
            return jsonObject;
        }
    }

authors的序列化由context来完成。context将会序列化给出的对象，并返回一个JsonElement。同时context也会尝试找到一个可以序列化当前对象的序列化器。如果没有找到，其将会使用默认的序列化器。

## 反序列化

    {
      'title':    'Java Puzzlers: Traps, Pitfalls, and Corner Cases',
      'isbn-10':  '032133678X',
      'isbn-13':  '978-0321336781',
      'authors':  ['Joshua Bloch', 'Neal Gafter']
    }

上面的Json对象包括4个字段，其中一个是数组。默认情况下，Gson期望Java类中的变量名与Json查找到的名称一样。因此，我们需要包含如下域名的类：title、isbn-10、isbn-13和authors。但是Java变量名不能包含减号。在接下来的实例中看到如何使用JsonDeserializer完全控制Json的解析[Gson注解实例](http://www.javacreed.com/gson-annotations-example/)。

    class Book {
        private String[] authors;
        private String isbn10;
        private String isbn13;
        private String title;
    }
    public class BookDeserializer implements JsonDeserializer<Book> {
        @Override
        public Book deserialize(final JsonElement json, final Type typeOfT, final JsonDeserializationContext context) throws JsonParseException {
            final JsonObject jsonObject = json.getAsJsonObject();
            JsonElement jsonTitle = jsonObject.get("title");
            final String title = jsonTitle.getAsString();
            final String isbn10 = jsonObject.get("isbn-10").getAsString();
            final String isbn13 = jsonObject.get("isbn-13").getAsString();
            final JsonArray jsonAuthorsArray = jsonObject.get("authors").getAsJsonArray();
            final String[] authors = new String[jsonAuthorsArray.size()];
            for (int i = 0; i < authors.length; i++) {
                final JsonElement jsonAuthor = jsonAuthorsArray.get(i);
                authors[i] = jsonAuthor.getAsString();
            }
            final Book book = new Book();
            book.setTitle(title);
            book.setIsbn10(isbn10);
            book.setIsbn13(isbn13);
            book.setAuthors(authors);
            return book;
        }
    }

接下来将会描述怎么反序列化嵌套对象。JsonDeserializer提供了一个JsonDeserializationContext实例作为`deserialize()`方法的第三个参数。我们可以将对象的反序列化委托给指定的JsonDeserializationContext实例。它将反序列化给定的JsonElement并返回一个指定类型的实例，代码如下：

    Author author = context.deserialize(jsonElement, Author.class);
    Author[] authors = context.deserialize(jsonObject.get("authors"), Author[].class);

## 参考资料：
- [完全理解Gson_2：Gson序列化](http://www.importnew.com/16638.html)
- [完全理解Gson_3：Gson反序列化](http://www.importnew.com/16786.html)