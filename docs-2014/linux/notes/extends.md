title: Linux | Science Lab Extends
date: 2017-07-04
tags: [Linux,CentOS,Ubuntu]
---
防火墙，时区修正，Resilio sync，R，Python，Pandoc，Auto.sh，Mysql。

<!--more-->
## 查看文件大小
```bash
df -h
du -h --max-depth=1 /data
```

## Firewall
```bash
# CentOS 7
systemctl stop firewalld.service
systemctl disable firewalld.service
# Ubuntu 16.04
sudo ufw disable
```

## Date
```bash
tzselect
# 5(Asia) -> 9(China) -> 1(Beijing Time) -> 1(Yes)
date
```

## Resilio sync

### yum
`vim /etc/yum.repos.d/resilio-sync.repo`输入内容：
```
[resilio-sync]
name=Resilio Sync
baseurl=http://linux-packages.resilio.com/resilio-sync/rpm/$basearch
enabled=1
gpgcheck=1
```

添加公钥及安装：
```bash
rpm --import https://linux-packages.resilio.com/resilio-sync/key.asc
yum update
yum -y install resilio-sync
```

### apt
注册Resilio存储库，命令：
```
echo "deb http://linux-packages.resilio.com/resilio-sync/deb resilio-sync non-free" | sudo tee /etc/apt/sources.list.d/resilio-sync.list
```

添加公钥及安装：
```
wget -qO - https://linux-packages.resilio.com/resilio-sync/key.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get install resilio-sync
```

### 配置服务
```
mkdir ~/work_main
systemctl enable resilio-sync
useradd -m -s /bin/bash hejian
passwd hejian
usermod -aG hejian rslsync
usermod -aG rslsync hejian
service resilio-sync start
```

浏览器输入`http://localhost:8888/gui/`，打开WebGUI设置同步特性。参考[Installing Sync package on Linux](https://help.getsync.com/hc/en-us/articles/206178924)和[Guide to Linux, and Sync peculiarities](https://help.getsync.com/hc/en-us/articles/204762449-Guide-to-Linux)。

### Remove Package
For Debian-based Linux：
```
sudo apt-get purge resilio-sync
```

For RPM-based Linux：
```
sudo yum remove resilio-sync
```

### 离线安装
Resilio的官方资源大陆不友好,无奈之下只有选择离线安装:
```
wget http://internal.resilio.com/2.5.5/resilio-sync_x64.tar.gz
tar zxf resilio-sync_x64.tar.gz
./rslsync --help
```

启动服务:
```
./rslsync --webui.listen 0.0.0.0:8888
```

看到`Resilio Sync forked to background. pid = 12085`字样表示成功,浏览器访问[ip:8888](#)进行配置与管理,查看进程`ps aux | grep -nE 'rslsync'`,停止服务`kill 12085`.

### FAQ
有时可能会出现状态OK，但是图标仍然在转的情况，或者不时闪现有任务要提交。多半是权限问题：
```
ll -Ra   ~/work_main | grep -nE "\\s+root\\s+" > tmp.txt
ls -RAlp ~/work_main | grep -nE "\\s+root\\s+"
ls -RAlp ~/work_main | grep -nE "\\s+hejian\\s+"
```

简单粗暴的解决方法：(也不能排除：隐藏文件的可能)
```
sudo chmod -R 770 ~/work_main
sudo chown -R rslsync:rslsync ~/work_main
```

## R

### yum
Install R:
```bash
yum -y install epel-release
yum update && yum info R && yum list R
yum list installed | grep -nE "^R"
yum -y install R
yum -y remove R
```

[RStudio Server](https://www.rstudio.com/products/rstudio/download-server/)：
```bash
wget https://download2.rstudio.org/rstudio-server-rhel-1.0.143-x86_64.rpm
sudo yum install --nogpgcheck rstudio-server-rhel-1.0.143-x86_64.rpm

wget https://download1.rstudio.org/rstudio-1.0.143-x86_64.rpm
sudo yum install --nogpgcheck rstudio-1.0.143-x86_64.rpm
```

浏览器中输入`http://127.0.0.1:8787`，用非root账号登录。

### Ubuntu 16.04
`sudo vim /etc/apt/sources.list`:
```
deb https://cloud.r-project.org/bin/linux/ubuntu xenial/
```

注意,`xenial/`由系统版本决定,我是`Ubuntu 16.04`.[link](https://cran.rstudio.com/bin/linux/ubuntu/README.html)

To install:
```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
sudo apt-get update
sudo apt-get install r-base
```

[RStuido Server](https://www.rstudio.com/products/rstudio/download-server/):
```
sudo apt-get install gdebi-core

wget https://download2.rstudio.org/rstudio-server-1.0.143-amd64.deb
sudo gdebi rstudio-server-1.0.143-amd64.deb

wget https://download1.rstudio.org/rstudio-1.0.143-amd64.deb
sudo gdebi rstudio-1.0.143-amd64.deb
```

浏览器中输入`http://127.0.0.1:8787`,用非root账号登录.

## Python
```bash
wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
bash Anaconda3-4.3.1-Linux-x86_64.sh
echo $'export PATH="/root/anaconda3/bin:$PATH"' >> /etc/profile
source /etc/profile
conda info
conda update conda
```

你可能需要`py35`而不是默认的`py36`，比如`keras`还不支持`py36`：
```
conda install python=3.5
```

管理包：
```bash
conda list | grep -nE "^pillow"
conda search package_name
conda install package_name
conda remove package_name
rm -rf ~/anaconda3
```

管理环境：
```bash
conda info -e #列出环境
conda create -n lab_py2 python=2 #新建环境
Windows: activate lab_py2 #激活环境
Windows: deactivate lab_py2 #停用环境
Linux: source activate lab_py2 #激活环境
Linux: source deactivate lab_py2 #停用环境
conda remove -n lab_py2 --all #删除环境
```

## Pandoc
[Installing pandoc](http://pandoc.org/installing.html)

## Auto.sh
`vim /root/sh_start.sh`输入内容：
```bash
#!/bin/bash
echo "sh_start :: jupyter :: $(date) .." >> /root/sh_start_log.txt
su - root -c "{需要执行的指令} &"
echo ".." >> /root/sh_start_log.txt
```

添加可执行的权限：
```bash
chmod +x /root/sh_start.sh
```

`vim /etc/rc.local`添加内容：
```bash
echo "rc.local :: $(date) .." >> /root/sh_start_log.txt
sudo bash /root/sh_start.sh
```

添加可执行的权限：
```bash
chmod +x /etc/rc.local
```

## Jupyter-notebook
设置开机启动,在`/etc/rc.local`添加:
```
export JAVA_HOME=/home/hejian/apps/jdk1.8.0_161
cd /root/apps/
/root/anaconda3/bin/python python-jupyter-notebook.py --allow-root --NotebookApp.iopub_data_rate_limit=0 --ip=127.0.0.1 --port=9091 --no-browser --notebook-dir /data1/hej &
```

`/root/apps/python-jupyter-notebook.py`内容如下:
```
if __name__ == '__main__':
    import sys
    import notebook.notebookapp
    sys.exit(notebook.notebookapp.main())
```

免密码登录:
```bash
$ jupyter-notebook --generate-config --allow-root
$ sudo vim /root/.jupyter/jupyter_notebook_config.py
#-----------------------------------------------------
## new line
c.NotebookApp.token = ''
#-----------------------------------------------------
```

## Mysql
[Installing and Upgrading](https://dev.mysql.com/doc/refman/5.7/en/installing.html)

### yum
[mysql57-community-release-el7-9.noarch.rpm](https://dev.mysql.com/doc/mysql-yum-repo-quick-guide/en/):
```bash
wget https://repo.mysql.com//mysql57-community-release-el7-9.noarch.rpm
sudo rpm -Uvh mysql57-community-release-el7-9.noarch.rpm
yum repolist all | grep mysql
sudo yum install mysql-community-server

sudo systemctl start mysqld.service
sudo systemctl status mysqld.service
# 查看随机密码
sudo grep 'temporary password' /var/log/mysqld.log
# 开启远程访问
mysql -uroot -p
mysql> show variables like '%character%';
mysql> use mysql;
mysql> update user set authentication_string=password('root') where user='root';
mysql> update user set host='%' where user='root';
mysql> flush privileges;
mysql> quit
```

修改字符编码，`vim /etc/my.cnf`：
```
[mysqld]
datadir=/var/lib/mysql
socket=/var/lib/mysql/mysql.sock
character-set-server=utf8
```

### apt
[mysql-apt-config_0.8.6-1_all.deb](https://dev.mysql.com/doc/mysql-apt-repo-quick-guide/en/):
```
sudo dpkg -i mysql-apt-config_0.8.6-1_all.deb
sudo dpkg-reconfigure mysql-apt-config

sudo apt-get update
sudo apt-get install mysql-server

sudo service mysql status
sudo service mysql stop
sudo service mysql start

sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf
## add: character-set-server=utf8
## del: bind-address    = 127.0.0.1
sudo service mysql restart
```

Login:
```
mysql -uroot -proot
mysql> show variables like '%character%';
mysql> use mysql;
mysql> update user set authentication_string=password('root') where user='root';
mysql> update user set host='%' where user='root';
mysql> flush privileges;
mysql> quit
mysql --user=root --password=root
mysql --host=192.168.31.217 --user=root --password=root
```

## 科学上网
squid:
```
sudo apt update
sudo apt install squid
```

`sudo vim /etc/squid/squid.conf`,完成配置项:
```
http_access deny all
  => @line: 1190
http_access allow all

http_port 3128
  => @line: 1599
http_port 47.52.128.170:3128
```

服务器端要监听的端口,默认3128,可以不改,浏览器设置代理时要用.然后,重启服务:
```
sudo service squid restart
```

[Squid + Pac代理](https://www.deadend.me/2016/07/29/cross-the-gfw/)