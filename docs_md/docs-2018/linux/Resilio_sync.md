# Resilio sync

## yum
`vim /etc/yum.repos.d/resilio-sync.repo`:
```
[resilio-sync]
name=Resilio Sync
baseurl=http://linux-packages.resilio.com/resilio-sync/rpm/$basearch
enabled=1
gpgcheck=1
```

添加公钥及安装:
```bash
rpm --import https://linux-packages.resilio.com/resilio-sync/key.asc
yum update
yum -y install resilio-sync
```

## apt
注册Resilio存储库:
```bash
echo "deb http://linux-packages.resilio.com/resilio-sync/deb resilio-sync non-free" | sudo tee /etc/apt/sources.list.d/resilio-sync.list
```

添加公钥及安装:
```bash
wget -qO - https://linux-packages.resilio.com/resilio-sync/key.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get install resilio-sync
```

## 配置服务
```bash
useradd hejian -s /bin/bash -g rslsync
passwd hejian
mkdir ~/work_main
systemctl enable resilio-sync
usermod -aG hejian rslsync
service resilio-sync start
```

浏览器输入`http://localhost:8888/gui/`,打开WebGUI设置同步特性.参考[Installing Sync package on Linux](https://help.getsync.com/hc/en-us/articles/206178924)和[Guide to Linux, and Sync peculiarities](https://help.getsync.com/hc/en-us/articles/204762449-Guide-to-Linux).

## Remove Package
For Debian-based Linux：
```
sudo apt-get purge resilio-sync
```

For RPM-based Linux：
```
sudo yum remove resilio-sync
```

## 离线安装
Resilio的官方资源大陆不友好,无奈之下只有选择离线安装:
```
wget http://internal.resilio.com/2.5.5/resilio-sync_x64.tar.gz
tar zxf resilio-sync_x64.tar.gz
./rslsync --help
```

启动服务:
```bash
./rslsync --webui.listen 0.0.0.0:8888
```

看到`Resilio Sync forked to background. pid = 12085`字样表示成功,浏览器访问[ip:8888](#)进行配置与管理,查看进程`ps aux | grep -nE 'rslsync'`,停止服务`kill -9 12085`.浏览器打开`http://host-ip:8888/gui/`.添加到开机启动:
```
# sudo vim /etc/rc.local
cd /data/apps
su - username -c "/data/apps/rslsync --webui.listen 0.0.0.0:8888 &"
```

## FAQ
图标一直转,或不时闪现有任务要提交.多半是权限问题:
```bash
ll -Ra   ~/work_main | grep -nE "\\s+root\\s+" > tmp.txt
ls -RAlp ~/work_main | grep -nE "\\s+root\\s+"
ls -RAlp ~/work_main | grep -nE "\\s+hejian\\s+"
```

简单粗暴的解决方法:(不排除隐藏文件的可能)
```bash
sudo chmod -R 770 ~/work_main
sudo chown -R rslsync:rslsync ~/work_main
```
