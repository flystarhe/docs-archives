# Nohub
本文内容包括:后台运行和端口检查.`nohup`不挂断地运行命令,`&`在后台运行.

`&`的意思是忽略SIGINT信号,所以当你运行`command &`的时候,使用`ctrl+c`不会结束进程.但是要注意,如果你关掉shell后,进程会消失.

`nohup`的意思是忽略SIGHUP信号,所以当你运行`nohup command`的时候,关闭shell不会结束进程.但是要注意,如果你直接在shell中用`ctrl+c`,进程会消失.

## nohup
无论是否将`nohup`命令的输出重定向到终端,输出都将附加到当前目录的`nohup.out`文件中.如果当前目录的`nohup.out`文件不可写,输出重定向到`$HOME/nohup.out`文件中.

一般两个一起用`nohup command &`,举例:
```bash
nohup python app.py > tmps/log.app.py00 2>&1 &
```

在上面的例子中:

- 0: stdin(standard input)
- 1: stdout(standard output)
- 2: stderr(standard error)

`2>&1`是将标准错误`2`重定向到标准输出`&1`,标准输出`&1`再被重定向输入到`tmps/log.app`文件中.

## 端口检查
根据进程查端口:
```bash
sudo lsof -i | grep {pid}
sudo netstat -nap | grep {pid}
```

根据端口查进程:
```bash
sudo lsof -i:{port}
sudo netstat -nap | grep {port}
```

终止后台运行的进程:
```bash
kill -9 {pid}
```
