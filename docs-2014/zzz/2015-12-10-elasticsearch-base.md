title: Elasticsearch入门
date: 2015-12-10
tags: [CentOS,Elasticsearch]
---
Elasticsearch是一个实时分布式搜索和分析引擎。我们还能这样去描述它：分布式的实时文件存储，每个字段都被索引并可被搜索；分布式的实时分析搜索引擎；可以扩展到上百台服务器，处理PB级结构化或非结构化数据。

<!--more-->
## 安装
理解Elasticsearch最好的方式是去运行它，安装Elasticsearch唯一的要求是安装官方新版的Java，你可以从`elasticsearch.org/download`下载最新版本的Elasticsearch。

    $ su worker
    $ wget https://download.elasticsearch.org/elasticsearch/release/org/elasticsearch/distribution/zip/elasticsearch/2.1.0/elasticsearch-2.1.0.zip
    $ unzip elasticsearch-2.1.0.zip
    $ cd elasticsearch-2.1.0
    $ ./bin/elasticsearch

打开另一个终端进行测试：

    // 查看ELasticsearch集群已经启动情况
    $ curl 'http://localhost:9200/?pretty'
    // 计算集群中的文档数量
    $ curl -i -XGET 'http://localhost:9200/_count?pretty' -d '
    {
        "query": {
            "match_all": {}
        }
    }'

## 索引
在Elasticsearch中，文档归属于一种类型(type)，而这些类型存在于索引(index)中，我们可以画一些简单的对比图来类比传统关系型数据库：

    Relational DB -> Databases -> Tables -> Rows      -> Columns
    Elasticsearch -> Indices   -> Types  -> Documents -> Fields

它不需要额外的管理操作，比如创建索引或者定义每个字段的数据类型。我们能够直接索引文档，因为Elasticsearch已经内置所有的缺省设置。

    // megacorp:索引(数据库);employee:类型(表);1:文档(id);
    // 指定id PUT
    $ curl -i -XPUT 'http://localhost:9200/megacorp/employee/1?pretty' -d '{"first_name": "John","last_name": "Smith","age": 25,"about": "I love to go rock climbing","interests": ["sports","music"]}'
    // 自增id PUT >> POST
    $ curl -i -XPOST 'http://localhost:9200/megacorp/employee/?pretty' -d '{"first_name": "Jane","last_name": "Smith","age": 33,"about": "I like to go rock climbing","interests": ["music","code"]}'

事实上，可以通过在`config/elasticsearch.yml`中添加配置`action.auto_create_index: false`来防止自动创建索引。一个比较全面的索引创建如下：

    curl -i -XPUT 'http://localhost:9200/my_index_1?pretty' -d '
    {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "my_type_1": {
                "_id": {
                    "path": "post_id",
                    "store": false,
                    "index": "not_analyzed"
                },
                "_type": {
                    "store": true,
                    "index": "no"
                },
                "_source": {
                    "enabled": true,
                    "includes": ["s1"],
                    "excludes": ["s2"]
                },
                "_all": {
                    "enabled": true
                },
                "include_in_all": false,
                "properties": {
                    "s1": {
                        "type": "string",
                        "store": true,
                        "include_in_all": true,
                        "index": "analyzed"
                    },
                    "s2": {
                        "type": "string",
                        "store": false,
                        "include_in_all": false,
                        "index": "analyzed"
                    }
                }
            }
        }
    }'

我们可以使用`_mapping`后缀来查看Elasticsearch中的映射。例如索引`megacorp`类型`employee`中的映射：

    $ curl -i -XGET 'http://localhost:9200/megacorp/_mapping/employee?pretty'

你可以为已有的类型更新映射。我们决定在`employee`的映射中增加一个新的`not_analyzed`类型的文本字段，叫做`tag`，使用`_mapping`后缀：

    $ curl -i -XPUT 'http://localhost:9200/megacorp/_mapping/employee?pretty' -d '
    {
        "properties": {
            "tag": {
                "type": "string",
                "index": "not_analyzed"
            }
        }
    }'

## 搜索
检索文档，执行`GET`请求并指出文档的“地址”既可返回原始JSON文档，使用`HEAD`检查文档是否存在。

    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/1?pretty'
    $ curl -i -XHEAD 'http://localhost:9200/megacorp/employee/1?pretty'

简单搜索：

    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty'
    // 检索姓为Smith的
    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty&q=last_name:Smith'
    // 检索文档的一部分
    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty&_source=interests'
    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty' -d '
    {
        "query": {
            "match": {
                "last_name": "Smith"
            }
        }
    }'

复杂搜索：依旧想要找到姓氏为“Smith”的员工，还想年龄大于30岁的员工。

    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty' -d '
    {
        "query": {
            "filtered": {
                "filter": {
                    "range": {
                        "age" : {"gt": 30}
                    }
                },
                "query": {
                    "match": {
                        "last_name": "Smith"
                    }
                }
            }
        }
    }'

全文搜索：传统数据库很难实现的功能。

    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty' -d '
    {
        "query": {
            "match": {
                "about": "rock climbing"
            }
        }
    }'

定制搜索：很多时候仅需要反馈特定某些字段。

    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty' -d '
    {
        "fields": ["about","interests"],
        "query": {
            "match": {
                "last_name": "Smith"
            }
        }
    }'

其他，如短语搜索、高亮等内容这里就略过了。

## 索引管理
索引只是一个用来指向一个或多个分片(shards)的“逻辑命名空间”。把分片想象成数据的容器。文档存储在分片中，然后分片分配到你集群中的节点上。分片可以是主分片或者是复制分片。你索引中的每个文档属于一个单独的主分片，所以主分片的数量决定了索引最多能存储多少数据。复制分片只是主分片的一个副本。创建分配3个主分片和1个复制分片的`blogs`索引如下：

    $ curl -i -XPUT 'http://localhost:9200/blogs?pretty' -d '
    {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 1
        }
    }'

主分片的数量在创建索引时已经确定。然而，主分片或者复制分片都可以处理读请求，所以数据的冗余越多，我们能处理的搜索吞吐量就越大。复制分片的数量可以在运行中的集群中动态地变更，让我们把复制分片的数量从原来的1增加到2：

    $ curl -i -XPUT 'http://localhost:9200/blogs/_settings?pretty' -d '
    {
        "number_of_replicas": 2
    }'

## 聚合分析
Elasticsearch有个功能叫做聚合(aggregations)，它允许你在数据上生成复杂的分析统计。它很像SQL中的`GROUP BY`但是功能更强大。举个例子，让我们找到所有职员中最大的共同点(兴趣爱好)是什么：

    $ curl -i -XGET 'http://localhost:9200/megacorp/employee/_search?pretty' -d '
    {
        "query": {
            "match": {
                "last_name": "Smith"
            }
        },
        "aggs": {
            "all_interests": {
                "terms": {
                    "field": "interests"
                }
            }
        }
    }'

## 关于集群
一个节点(node)就是一个Elasticsearch实例，而一个集群(cluster)由一个或多个节点组成，它们具有相同的`cluster.name`(请看`./config/elasticsearch.yml`)，它们协同工作，分享数据和负载。当加入新的节点或者删除一个节点时，集群就会感知到并平衡数据。在Elasticsearch集群中可以监控统计很多信息，集群健康是最重要的，集群健康有三种状态：green、yellow或red。

    $ curl -i -XGET 'http://localhost:9200/_cluster/health?pretty'

## 参考资料：
- [Elasticsearch权威指南_中文版](https://www.gitbook.com/book/looly/elasticsearch-the-definitive-guide-cn/details)