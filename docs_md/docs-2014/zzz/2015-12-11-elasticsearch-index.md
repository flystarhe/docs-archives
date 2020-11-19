title: Elasticsearch深入索引
date: 2015-12-11
tags: [CentOS,Elasticsearch,Index]
---
我们已经看到Elasticsearch如何在不需要任何预先计划和设置的情况下，轻松地开发一个新的应用。它包含几乎所有的和索引及类型相关的定制选项。在这里，将介绍管理索引和类型映射的API以及最重要的设置。

<!--more-->
## 创建索引
迄今为止，我们简单的通过添加一个文档的方式创建了一个索引。这个索引使用默认设置，新的属性通过动态映射添加到分类中。现在我们手动创建索引，在请求中加入所有设置和类型映射，如下所示：

    $ curl -i -XPUT 'http://localhost:9200/my_index?pretty' -d '
    {
        "settings": { ...any mappings... },
        "mappings": {
            "type_one": { ...any mappings...},
            "type_two": { ...any mappings...},
            ...
        }
    }'

Elasticsearch提供了优化好的默认配置。除非你明白这些配置的行为和为什么要这么做，请不要修改这些配置。下面是两个最重要的设置：

    number_of_shards   主分片个数，默认值是 5。索引创建后不能修改。
    number_of_replicas 复制分片个数，默认是 1。可以随时在活跃的索引上修改。

事实上，可以通过在`config/elasticsearch.yml`中添加配置`action.auto_create_index: false`来防止自动创建索引。用【索引模板】来自动预先配置索引。这在索引日志数据时尤其有效：你将日志数据索引在一个以日期结尾的索引上，第二天，一个新的配置好的索引会自动创建好。

## 删除索引
使用以下的请求来删除单个索引，多个索引，使用通配符，甚至所有索引。

    $ curl -i -XDELETE 'http://localhost:9200/my_index?pretty'
    $ curl -i -XDELETE 'http://localhost:9200/index_one,index_two?pretty'
    $ curl -i -XDELETE 'http://localhost:9200/index_*?pretty'
    $ curl -i -XDELETE 'http://localhost:9200/_all?pretty'

## 配置分析器
索引设置的analysis部分，用来配置已存在的分析器或创建自定义分析器来定制化你的索引。standard分析器是用于全文字段的默认分析器，对于大部分西方语系来说是一个不错的选择。它考虑了以下几点：

- standard 分词器，在词层级上分割输入的文本。
- standard 表征过滤器，被设计用来整理分词器触发的所有表征。
- lowercase 表征过滤器，将所有表征转换为小写。
- stop 表征过滤器，删除所有可能会造成搜索歧义的停用词，如a，the，and，is。

默认情况下，停用词过滤器是被禁用的。如需启用它，你可以通过创建一个基于`standard`分析器的自定义分析器，并且设置`stopwords`参数。可以提供一个停用词列表，或者使用一个特定语言的预定停用词列表。创建了一个新的分析器，叫做`es_std`，并使用预定义的西班牙语停用词：

    $ curl -i -XPUT 'http://localhost:9200/spanish_docs?pretty' -d '
    {
        "settings": {
            "analysis": {
                "analyzer": {
                    "es_std": {
                        "type":      "standard",
                        "stopwords": "_spanish_"
                    }
                }
            }
        }
    }'

`es_std`分析器不是全局的，它仅仅存在于我们定义的`spanish_docs`索引中。为了用`analyze API`来测试它，我们需要使用特定的索引名。(结果中显示停用词El被正确的删除了)

    $ curl -i -XGET 'http://localhost:9200/spanish_docs/_analyze?pretty&analyzer=es_std' -d 'El veloz zorro'

## 自定义分析器
分析器是三个顺序执行的组件的结合(字符过滤器，分词器，表征过滤器)。配置一个这样的分析器：

1. 用 html_strip 字符过滤器去除所有的 HTML 标签；
2. 将 & 替换成 and，使用一个自定义的 mapping 字符过滤器；
3. 使用 standard 分词器分割单词，使用 lowercase 表征过滤器将词转为小写，用 stop 表征过滤器去除一些自定义停用词。

    $ curl -i -XPUT 'http://localhost:9200/my_index?pretty' -d '
    {
        "settings": {
            "analysis": {
                "char_filter": {
                    "&_to_and": {
                        "type": "mapping",
                        "mappings": ["& => and"]
                    }
                },
                "filter": {
                    "my_stopwords": {
                        "type": "stop",
                        "stopwords": ["the","a"]
                    }
                },
                "analyzer": {
                    "my_analyzer": {
                        "type":        "custom",
                        "char_filter": ["html_strip","&_to_and"],
                        "tokenizer":   "standard",
                        "filter":      ["lowercase","my_stopwords"]
                    }
                }
            }
        }
    }'

测试新的分析器：

    $ curl -i -XGET 'http://localhost:9200/my_index/_analyze?pretty&analyzer=my_analyzer' -d 'The <em>quick</em> & brown fox'

除非我们告诉Elasticsearch在哪里使用，否则分析器不会起作用。我们可以通过下面的映射将它应用在一个string类型的字段上：

    $ curl -i -XPUT 'http://localhost:9200/my_index/_mapping/my_type?pretty' -d '
    {
        "properties": {
            "title": {
                "type": "string",
                "analyzer": "my_analyzer"
            }
        }
    }'

## 类型和映射
映射是Elasticsearch将复杂JSON文档映射成Lucene需要的扁平化数据的方式。例如，user类型中name字段的映射声明这个字段是一个string类型，在被加入倒排索引之前，它的数据需要通过whitespace分析器来分析。

    "name": {
        "type": "string",
        "analyzer": "whitespace"
    }

## 文档ID
文档唯一标识由四个元数据字段组成：

    `_id` ：文档的字符串`ID`
    `_type` ：文档的类型名
    `_index` ：文档所在的索引
    `_uid` ：`_type`和`_id`连接成的`type#id`

`_id`字段有一个你可能用得到的设置`path`设置告诉Elasticsearch它需要从文档本身的哪个字段中生成`_id`：

    PUT /my_index
    {
        "mappings": {
            "my_type": {
                "_id": {
                    "path": "doc_id"
                },
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "index": "not_analyzed"
                    }
                }
            }
        }
    }

从`doc_id`字段生成`_id`。警告：虽然这样很方便，但是注意它对bulk请求有个轻微的性能影响。处理请求的节点将不能仅靠解析元数据行来决定将请求分配给哪一个分片，而需要解析整个文档主体。

## 动态映射
或许你不知道今后会有哪些字段加到文档中，但是你希望它们能自动被索引。或许你仅仅想忽略它们。特别是当你使用Elasticsearch作为主数据源时，你希望未知字段能抛出一个异常来警示你。你可以通过`dynamic`设置来控制这些行为，它接受下面几个选项：

    true ：自动添加字段(默认)
    false ：忽略字段
    strict ：当遇到未知字段时抛出异常

`dynamic`设置可以用在根对象或任何`object`对象上。你可以将`dynamic`默认设置为`strict`，而在特定内部对象上启用它：(当遇到未知字段时，`my_type`对象将会抛出异常，`stash`对象会自动创建字段)

    PUT /my_index
    {
        "mappings": {
            "my_type": {
                "dynamic": "strict",
                "properties": {
                    "title": { "type": "string"},
                    "stash": {
                        "type": "object",
                        "dynamic": true
                    }
                }
            }
        }
    }

## 动态模板
使用`dynamic_templates`，你可以完全控制新字段的映射，你设置可以通过字段名或数据类型应用一个完全不同的映射。模板按照顺序来检测，第一个匹配的模板会被启用。例如，我们给string类型字段定义两个模板`es : 字段名以 _es 结尾需要使用 spanish 分析器`和`en : 所有其他字段使用 english 分析器`：

    PUT /my_index
    {
        "mappings": {
            "my_type": {
                "dynamic_templates": [
                    { "es": {
                        "match": "*_es",
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "string",
                            "analyzer": "spanish"
                        }
                    }},
                    { "en": {
                        "match": "*",
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "string",
                            "analyzer": "english"
                    }
                    }}
                ]
            }
        }
    }

`match_mapping_type`限制模板只能使用在特定的类型上，`match`参数只匹配字段名。

## 默认映射
通常，一个索引中的所有类型具有共享的字段和设置。用`_default_`映射来指定公用设置会更加方便，而不是每次创建新的类型时重复操作。`_default_`映射像新类型的模板。所有在`_default_`映射之后的类型将包含所有的默认设置，除非在自己的类型映射中明确覆盖这些配置。我们可以使用`_default_`映射对所有类型禁用`_all`字段，而只在`blog`字段上开启它：

    PUT /my_index
    {
        "mappings": {
            "_default_": {
                "_all": { "enabled": false }
            },
            "blog": {
                "_all": { "enabled": true }
            }
        }
    }

### 重建索引
修改在已存在的数据最简单的方法是重新索引：创建一个新配置好的索引，然后将所有的文档从旧的索引复制到新的上。`_source`字段的一个最大的好处是你已经在Elasticsearch中有了完整的文档，你不再需要从数据库中重建你的索引。你可以在同一时间执行多个重新索引的任务，但是你显然不愿意它们的结果有重叠。所以，可以将重建大索引的任务通过日期或时间戳字段拆分成较小的任务：

    GET /old_index/_search?search_type=scan&scroll=1m
    {
        "query": {
            "range": {
                "date": {
                    "gte": "2014-01-01",
                    "lt": "2014-02-01"
                }
            }
        },
        "size": 1000
    }

为了更高效的索引旧索引中的文档，使用`scan-scoll`来批量读取旧索引的文档，然后将通过`bulk API`来将它们推送给新的索引。

### 索引别名
重新索引过程中的问题是必须更新你的应用，来使用另一个索引名。索引别名正是用来解决这个问题的！索引别名就像一个快捷方式或软连接，可以指向一个或多个索引，也可以给任何需要索引名的API使用。别名带给我们极大的灵活性，允许我们做到：

    在一个运行的集群上无缝的从一个索引切换到另一个
    给多个索引分类，例如：last_three_months
    给索引的一个子集创建视图

创建一个索引`my_index_v1`，然后将别名`my_index`指向它：

    PUT /my_index_v1
    PUT /my_index_v1/_alias/my_index
    GET /*/_alias/my_index      //检测这个别名指向哪个索引
    GET /my_index_v1/_alias/*   //检测哪些别名指向这个索引

别名可以指向多个索引，所以我们需要在新索引中添加别名的同时从旧索引中删除它。这个操作需要原子化，所以我们需要用`_aliases`操作：(从旧索引迁移到了新的，而没有停机时间)

    POST /_aliases
    {
        "actions": [
            { "remove": { "index": "my_index_v1", "alias": "my_index" }},
            { "add": { "index": "my_index_v2", "alias": "my_index" }}
        ]
    }

## 参考资料：
- [Elasticsearch权威指南_中文版](https://www.gitbook.com/book/looly/elasticsearch-the-definitive-guide-cn/details)