title: Linux | 进阶小抄
date: 2016-08-01
tags: [Linux]
---
Shell是用户与Linux操作系统沟通的桥梁。用户既可以输入命令执行，又可以利用Shell脚本编程，完成更加复杂的操作。深入地了解和熟练地掌握Shell，是每一个Linux用户的必修功课之一。

<!--more-->
## 获取命令/脚本执行时间
```
$ time pwd
real    0m0.079s
user    0m0.000s
sys     0m0.000s
$ export TIMEFORMAT="real time %E, user time %U,sys time %S"
$ time ll
real time 0.038, user time 0.000,sys time 0.002
$ time bash exec.sh
```

## 查看文件编码与转码
```
$ file -bi tmp.txt
text/plain; charset=iso-8859-1
$ iconv -f gbk -t utf-8  tmp.txt > tmp.txt.utf8
$ cat exec.sh
for i in `find ./ -type f -name '*.TXT'` ;
do
    echo $i
    echo ${i}.tmp
    iconv -f gbk -t utf-8 $i > ${i}.tmp
    mv ${i}.tmp $i;
done
```

## 进程管理
```
$ ps aux | grep ltp //查看
flume    26426  0.8  6.5 1227100 1058876 pts/0 Sl+  05:29   0:02 ./bin/ltp_server --port 9091
$ pgrep -l ltp //查看
26426 ltp_server
$ kill 26426 //进程id
$ pkill ltp_server //进程名
```

## 压缩和解压
```
$ zip -r mydata.zip mydata //压缩mydata目录
$ zip -r abc123.zip abc 123.txt //abc文件夹和123.txt压缩成为abc123.zip
$ unzip -l mydata.zip //查看内容结构
$ unzip mydata.zip -d mydatabak //mydata.zip解压到mydatabak目录里面
$ unzip wwwroot.zip //wwwroot.zip直接解压到当前目录里面
```

`*.7z` on Ubuntu:
```
$ sudo apt install p7zip-full
$ 7z a -t7z -r filename.7z test/*
$ 7z x filename.7z -r -otmp
```

- `a`代表添加文件/文件夹到压缩包
- `-t`是指定压缩类型,比如`7z`
- `-r`表示递归所有的子文件夹
- `x`代表按原始目录解压(`e`也是解压,但会将所有文件都解压到根下)
- `-o`是指定解压到的目录,注意与`tmp`之间没有空格

## cat&split/合并&切分
```
$ cat 1.txt 2.txt >> all.txt
合并1.txt和2.txt文件内容到all.txt
$ split -l 100 -d trainfile.txt filedir/
每100行数据为一个新文本存到filedir目录
```

## 不流氓的Shell脚本
这个脚本编写了一个日志函数`shell_log`，直接执行`shell_log`把日志内容当作第一个参数传给它就可以了。
```
#!/bin/bash

# Shell Env
SHELL_NAME="shell_template.sh"
SHELL_DIR="/opt/shell"
SHELL_LOG="${SHELL_DIR}/${SHELL_NAME}.log"
LOCK_FILE="/tmp/${SHELL_NAME}.lock"

#Write Log 
shell_log(){
    LOG_INFO=$1
    echo "$(date "+%Y-%m-%d") $(date "+%H-%M-%S") : ${SHELL_NAME} : ${LOG_INFO}" >> ${SHELL_LOG}
}

shell_log "shell beginning, Write log test"
shell_log "shell success, Write log test"
```

## 管理用户
新建用户:
```
$ sudo mkdir -p -m 777 /data/share/work-dir
$ sudo useradd new-user -s /bin/bash -g ori-group -d /data/share/work-dir
$ sudo passwd new-user
$ sudo usermod -aG docker new-user
$ sudo usermod -aG sudo new-user
$ sudo tail /etc/group
$ sudo tail /etc/passwd
$ sudo userdel new-user
```

`su user`和`su - user`都可以切换用户，区别是：

- `su user`切换到其他用户，但是不切换环境变量；
- `su - user`则是完整的切换到新的用户环境。

## 远程登录
安装`open ssh`：
```
sudo apt-get install openssh-server
```

`sudo vim /etc/ssh/sshd_config`:
```
#PermitRootLogin prohibit-password
PermitRootLogin yes
```

重启服务：
```
service ssh restart
```

连接`ssh user@host -p port`。

## systemctl命令
启动nfs服务：
```
systemctl start nfs-server.service
```

设置开机自启动：
```
systemctl enable nfs-server.service
```

停止开机自启动：
```
systemctl disable nfs-server.service
```

查看服务状态：
```
systemctl status nfs-server.service
```

重新启动某服务：
```
systemctl restart nfs-server.service
```

查看所有已启动的服务：
```
systemctl list-units --type=service
```

## [用户名] is not in the sudoers file
切换到`root`用户,编辑`/etc/sudoers`:
```
$ su -
$ vim /etc/sudoers
root ALL=(ALL) ALL
[用户名] ALL=(ALL) ALL
```

如果提示只读,则执行`chmod u+w /etc/sudoers`,然后`chmod u-w /etc/sudoers`.

或者:
```
$ sudo usermod -aG sudo new-user
$ sudo usermod -aG admin new-user
```
