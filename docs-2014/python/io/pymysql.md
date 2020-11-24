title: 使用pymysql操作Mysql
date: 2017-08-25
tags: [Python]
---
pymysql是Python中操作Mysql的模块,使用方法与mysqldb相似,且支持py36,这个很关键.

<!--more-->
## 安装

    pip install pymysql

## 连接数据库
```
import pymysql

db_host = 'host address'
db_port = 3306
db_user = 'user name'
db_pass = 'pass word'
db_dbnm = 'db name'
conn = pymysql.connect(host=db_host, port=db_port,
                       user=db_user, password=db_pass,
                       database=db_dbnm, charset='utf8')
```

也可以使用字典,更优雅:
```
config = {'host': 'host address',
          'port': 3306,
          'user': 'user name',
          'password': 'pass word',
          'database': 'db name',
          'charset': 'utf8'}
conn = pymysql.connect(**config)
```

## 新建数据表
```
cursor = conn.cursor()
res_cnt = cursor.execute('drop table if exists proj_temp_test_flystar')
res_cnt = cursor.execute('create table if not exists proj_temp_test_flystar(id int,name text)')
cursor.close()
```

查看表结构:
```
cursor = conn.cursor()
res_cnt = cursor.execute('show create table proj_temp_test_flystar')
print(cursor.fetchall())
cursor.close()
```

输出:
```
(('proj_temp_test_flystar', 'CREATE TABLE `proj_temp_test_flystar` (\n  `id` int(11) DEFAULT NULL,\n  `name` text COLLATE utf8mb4_bin\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin'),)
```

## 插入数据
```
cursor = conn.cursor()
str_sql = "insert into proj_temp_test_flystar(id,name) values(%d,'%s')"
res_cnt = cursor.execute(str_sql % (1, 'name_1'))
res_cnt = cursor.execute(str_sql % (2, 'name_2'))
conn.commit()
cursor.close()
```

## 执行查询
```
cursor = conn.cursor()
str_sql = "select * from proj_temp_test_flystar"
res_cnt = cursor.execute(str_sql)
data = cursor.fetchall()# 取1/n行 fetchone()/fetchmany(3)
print(data)
cursor.close()
```

在fetch数据时按照顺序进行,可以使用`scroll(value, mode='relative')`来移动游标位置:
```
cursor.scroll(1, mode='relative')  # 相对当前位置移动
cursor.scroll(2, mode='absolute')  # 相对绝对位置移动
```

查询结果以字典的形式返回:
```
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
str_sql = "select * from proj_temp_test_flystar"
res_cnt = cursor.execute(str_sql)
data = cursor.fetchall()# 取1/n行 fetchone()/fetchmany(3)
print(data)
cursor.close()
```

## 参考资料:
- [pymssql examples](http://pymssql.org/en/latest/pymssql_examples.html)