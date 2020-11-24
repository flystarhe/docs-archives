# SFTP

## Install
```bash
# ubuntu
sudo apt-get install vsftpd
# centos
yum install ftp
yum install vsftpd
```

## useradd
```bash
sudo mkdir -p -m 755 /data/share/work-dir
sudo useradd iftp -s /bin/bash -g ftp -d /data/share/work-dir
sudo chown -R iftp:root /data/share/work-dir
sudo passwd iftp
sudo usermod -aG sudo iftp
sudo tail /etc/passwd
sudo tail /etc/group
sudo userdel iftp
sudo rm -rf /data/share/work-dir
```

## vsftpd.conf
```bash
# ubuntu: /etc/vsftpd.conf
# centos: /etc/vsftpd/vsftpd.conf
local_enable=YES
write_enable=YES
utf8_filesystem=YES
```

## Test
```bash
sftp iftp@192.168.31.197:/data/share
iftp@192.168.31.197's password:
Connected to 192.168.31.197.
Changing to: /data/share
sftp> exit
```

use ftp:
```bash
(base) [root@gpuserver81 hxgd]# ftp
ftp> open 182.150.44.163 2021
ftp> ls
ftp> cd CSOT-T6
ftp> lcd /data/hxgd
ftp> get 1.txt
ftp> exit
```

## Notes
```bash
# ubuntu
sudo service vsftpd stop
sudo service vsftpd start
sudo service vsftpd restart
# centos
systemctl start vsftpd.service
systemctl restart vsftpd.service
```
