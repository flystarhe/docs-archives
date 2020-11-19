title: Memcached安装配置说明书
date: 2015-01-13
tags: [CentOS,Memcached,PHP]
---
Memcached是一个高性能的分布式内存对象缓存系统，用于动态Web应用以减轻数据库负载。它通过在内存中缓存数据和对象来减少读取数据库的次数，从而提高动态、数据库驱动网站的速度。Memcached基于一个存储键值对的HashMap。

<!--more-->
## 准备PHP环境
网上有非常多关于`PHP`+`Nginx`+`Mysql`的教程，这里不再赘述，跳过。

## 安装SASL环境
执行`yum install cyrus-sasl-plain cyrus-sasl cyrus-sasl-devel cyrus-sasl-lib`，可以先用`rpm -qa | grep sasl`检查是否安装，若已安装则无需再安装。

## 安装libmemcached
```
wget https://launchpad.net/libmemcached/1.0/1.0.16/+download/libmemcached-1.0.16.tar.gz
tar -zxvf libmemcached-1.0.16.tar.gz
cd libmemcached-1.0.16
./configure --prefix=/usr/local/libmemcached --enable-sasl
make
make install
cd ..
```

## 安装memcached
安装memcached前需要确认是否有`zlib-devel`包，没有则需要执行`yum install zlib-devel`。
```
wget http://pecl.php.net/get/memcached-2.1.0.tgz
tar -zxvf memcached-2.1.0.tgz
cd memcached-2.1.0
phpize #如果系统中有两套PHP环境，需绝对路径调用该命令/usr/bin/phpize
./configure --with-libmemcached-dir=/usr/local/libmemcached --enable-memcached-sasl
make
make install
```

最后修改php.ini文件，增加：
```
extension=memcached.so
memcached.use_sasl = 1
```

>依赖: Memcached 2.1.0扩展必须使用libmemcached 1.0.x的库，低于1.0的库不再能够成功编译。编译libmemcached时GCC要求在4.2以上。

## 安装检查
重启Nginx PHP环境`/etc/init.d/php-fpm restart`（不一定非得重启，不过我当时调试不通是这样处理的），访问执行phpinfo，搜索memcached，若能搜索到就说明已经ok了。当然推荐使用如下代码测试是否部署成功，请修改代码中的地址、端口、用户名及密码：
```
<?php
$connect = new Memcached; //声明一个新的memcached链接
$connect->setOption(Memcached::OPT_COMPRESSION, false); //关闭压缩功能
$connect->setOption(Memcached::OPT_BINARY_PROTOCOL, true); //使用二进制协议
$connect->addServer('memcache-host', 11211); //地址及端口
$connect->setSaslAuthData('aaaaaaaaaa', 'password'); //连接帐号密码
$connect->set("hello", "world");
echo 'hello: ',$connect->get("hello");
$connect->quit();
?>
```
