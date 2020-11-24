# Supervisor
Supervisor是一个客户端/服务器系统,允许用户在类UNIX操作系统上控制大量进程.

Supervisor管理进程,是通过fork/exec的方式将这些被管理的进程当作supervisor的子进程来启动,所以我们只需要将要管理进程的可执行文件的路径添加到supervisor的配置文件中就好了.此时被管理进程被视为supervisor的子进程,若该子进程异常中断,则父进程可以准确的获取子进程异常中断的信息,通过在配置文件中设置`autostart=ture`,可以实现对异常中断的子进程的自动重启.

## 安装
安装方式很简单,执行`sudo apt-get install supervisor`即可.

也可以是:
```
git clone https://github.com/Supervisor/supervisor.git && \
cd supervisor && \
python setup.py install
```

指定版本(v3.3.4不支持py3):
```
wget https://github.com/Supervisor/supervisor/archive/3.3.4.tar.gz -O supervisor.tar.gz && \
tar zxf supervisor.tar.gz && \
cd supervisor-3.3.4 && \
python setup.py install
```

查看版本:
```
supervisord -v
## 4.0.0.dev0
```

## 配置文件
终端执行`echo_supervisord_conf`会打印一个配置文件样本.可以保存为文件:
```
echo_supervisord_conf > /etc/supervisord.conf
```

如果你没有root权限,或者你不想把配置文件存放在`/etc/supervisord.conf`.可以放在当前目录:
```
echo_supervisord_conf > supervisord.conf
```

## 添加程序
在supervisord为你做任何事情之前,你至少需要在其配置中添加一个`program`部分.编辑`supervisord.conf`:
```
;exists `/opt/log/`
;supervisord -c supervisord.conf -u hejian
;http://127.0.0.1:9001
;kill -s SIGTERM pid

[inet_http_server]         ; inet (TCP) server disabled by default
port=*:9001                ; ip_address:port specifier, *:port for all iface
username=user              ; default is no username (open server)
password=1234              ; default is no password (open server)

[supervisord]
logfile=/var/log/supervisor/supervisord.log ; (main log file;default $CWD/supervisord.log)
pidfile=/var/run/supervisord.pid            ; (supervisord pidfile;default supervisord.pid)
childlogdir=/var/log/supervisor             ; ('AUTO' child log dir, default $TEMP)

[supervisorctl]
serverurl=http://*:9001    ; use an http:// url to specify an inet socket

[program:jupyter_root]
command=/root/anaconda3/bin/jupyter-lab --allow-root --ip='*' --port=9002 --no-browser --notebook-dir='/data/root' --NotebookApp.token='hi'
directory=/data/root
environment=PATH="/root/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin"
user=root

[program:jupyter_ylab]
command=/root/anaconda3/bin/jupyter-lab --allow-root --ip='*' --port=9003 --no-browser --notebook-dir='/data/ylab' --NotebookApp.token='hi'
directory=/data/ylab
environment=PATH="/root/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin"
user=ylab
```

- command:该程序启动时将运行的命令
- directory:执行子进程时supervisord暂时切换到该目录
- user:指示supervisord将此用户帐户运行该程序
- startsecs:程序在启动后需要保持运行以考虑启动成功的总秒数
- redirect_stderr:如果是true,则进程的stderr输出被发送回其stdout
- stdout_logfile:将进程stdout输出到指定文件
- stdout_logfile_maxbytes:日志文件最大字节数,默认为50MB
- stdout_logfile_backups:日志文件的备份数量,默认为10

参考[Section Settings](http://supervisord.org/configuration.html#program-x-section-settings).

## 启动管理
```
/etc/init.d/supervisor stop
/etc/init.d/supervisor start
/etc/init.d/supervisor restart
```

或(不推荐,没有原因):
```
supervisord -c supervisord.conf -u root
```

或(不推荐,有时会不能工作):
```
/usr/bin/python /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
```

supervisor进程是不能直接杀死的,正确的方式是`kill -s SIGTERM pid`.