# Ubuntu
- [Ubuntu Releases](http://releases.ubuntu.com/)
- [如何在Ubuntu上创建可启动的U盘](https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-ubuntu#0)
- 推荐分区方式为`SSD(/,SWAP,ESP(EFI System Partition)), 机械硬盘(/home,/data)`

```
uname -a              # 查看内核/操作系统/CPU
head -n 1 /etc/issue  # 查看操作系统版本
env                   # 查看环境变量资源
uptime                # 查看系统运行时间/用户数/负载
w                     # 查看活动用户
id <username>         # 查看指定用户信息
last                  # 查看用户登录日志
crontab -l            # 查看当前用户的计划任务
```

## 管理用户
```
sudo mkdir -p -m 750 /data/share/work-dir
sudo useradd new-user -s /bin/bash -g ori-group -d /data/share/work-dir
sudo chown -R new-user:new-user /data/share/work-dir
sudo passwd new-user
sudo usermod -aG docker new-user
sudo usermod -aG sudo new-user
sudo tail /etc/group
sudo tail /etc/passwd
sudo userdel new-user
sudo rm -rf /data/share/work-dir
```

`su user`和`su - user`都可以切换用户,区别是:

- `su user`切换到其他用户,但是不切换环境变量
- `su - user`则是完整的切换到新的用户环境

命令`groupadd`创建用户组,创建`groupName`组,其GID为`9000`:
```
groupadd -g 9000 groupName
gpasswd -a user1 groupName
gpasswd -d user1 groupName
groupdel groupName
```

## 远程登录
安装`open ssh`:
```
sudo apt-get install openssh-server
```

`sudo vim /etc/ssh/sshd_config`:
```
#PermitRootLogin prohibit-password
PermitRootLogin yes
```

重启服务:
```
service ssh restart
```

连接`ssh user@host -p port`.

## 命令行和图形界面切换
从图形界面切换到命令行模式,我们可以通过按`CTRL+ALT+F1..F6`,为什么是`F1..F6`,因为在Linux中一般有F1到F6多个命令行字符终端,也就是说我们可以同时打开最多6个命令行界面.从命令行模式切换回图形界面,我们可以通过按`CTRL+ALT+F7`.

## 显卡驱动
注意:CUDA打包了显卡驱动,如果你确定要安装CUDA,且不在意驱动是否是最新版,可跳过此步骤.

对于使用NVIDIA的游戏玩家,可以下载[官方驱动程序](http://www.geforce.cn/drivers).图省事可以使用`sudo apt-get install nvidia-384`命令,也是推荐方法.

Ubuntu系统集成的显卡驱动程序是nouveau,它是第三方为NVIDIA开发的开源驱动,我们需要先将其屏蔽才能安装NVIDIA官方驱动.`sudo vim /etc/modprobe.d/blacklist.conf`添加黑名单:
```
blacklist nouveau
```

执行`sudo update-initramfs -u`更新,修改后需要重启系统.确认nouveau已经被你干掉,使用命令`lsmod | grep nouveau`.按`Ctrl+Alt+F1`切到命令行模式,关闭图形环境:
```
sudo service lightdm stop
```

先删除旧的驱动(完全卸载安装包,包括删除配置文件):
```
sudo apt-get purge nvidia*
```

安装驱动程序`NVIDIA-Linux-x86_64-384.130.run`:
```
lspci | grep VGA
# 00:0f.0 VGA compatible controller: VMware SVGA II Adapter
# 04:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] (rev a1)
# 0b:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] (rev a1)
# 13:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] (rev a1)
# 1b:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] (rev a1)
sudo sh NVIDIA-Linux-x86_64-384.130.run
```

重新启动图形环境:
```
sudo service lightdm start
```

按`Ctrl+Alt+F7`切到图形界面.用`nvidia-smi`命令查看显卡和驱动情况,列出GPU的信息列表则表示驱动安装成功.如果安装后驱动程序工作不正常,使用下面的命令进行卸载:
```
sudo sh NVIDIA-Linux-x86_64-384.130.run --uninstall
```

这时,你可能切到图形界面,在登录界面耗着,死活进不了图形界面.可能原因是你卸载了所有图形驱动,仅安装了独显驱动,但是你的主板的显卡初始选项确是集显(IGFX).所以,你需要重启,修改BIOS初始显卡选项为独显(PCIE).问题解决.

## CUDA 9.0
下载[cuda_9.0.176_384.81_linux.run](https://developer.nvidia.com/cuda-downloads),运行安装程序并按照屏幕提示操作:
```
sudo sh cuda_9.0.176_384.81_linux.run
```

如果你已经手动安装了合适的驱动,`Install NVIDIA Accelerated Graphics Driver`选择`no`.因为我们希望默认使用`CUDA8`,所以`symbolic link at /usr/local/cuda`选择`no`.执行卸载脚本即可卸载`CUDA`,默认路径为`/usr/local/cuda-9.0/bin`.

然后,配置环境`sudo vim /etc/profile`:
```
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
```

`sudo vim /etc/ld.so.conf`,添加:
```
/usr/local/cuda-9.0/lib64
```

执行`source /etc/profile`使生效,验证安装:
```
nvcc --version
```

安装补丁:
```
sudo sh cuda_9.0.176.1_linux.run
sudo sh cuda_9.0.176.2_linux.run
sudo sh cuda_9.0.176.3_linux.run
sudo sh cuda_9.0.176.4_linux.run
```

安装[cudnn-9.0-linux-x64-v7.0.4.tgz](https://developer.nvidia.com/cudnn):
```
tar -xzf cudnn-9.0-linux-x64-v7.0.4.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda-9.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64
```

执行`sudo ldconfig`,若提示`/usr/local/cuda-9.0/lib64/libcudnn.so.7`不是符号连接,则执行以下命令:
```
cd /usr/local/cuda-9.0/lib64
sudo mv libcudnn.so.7 libcudnn.so.7.old
sudo ln -s libcudnn.so.7.0.4 libcudnn.so.7
```

## 防火墙设置
```
ufw status
ufw enable
ufw default deny
ufw allow 9000
ufw delete allow 80
ufw allow from 192.168.1.1
ufw reload
ufw disable
```

## 搜狗输入法
下载搜狗拼音[sogoupinyin_2.2.0.0108_amd64.deb](https://pinyin.sogou.com/linux/),查看帮助转[help](https://pinyin.sogou.com/linux/help.php).

## 软件源镜像
Ubuntu的软件源配置文件是`/etc/apt/sources.list`.将系统自带的该文件做个备份,将该文件替换为下面内容,即可使用[TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)的软件源镜像:
```
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
```

## 添加全新硬盘
拆开机箱安装硬盘，开机查看系统分区情况：
```
sudo fdisk -l
```

可以看到新加的硬盘。接下里格式化硬盘，这里通过Ubuntu系统的disk软件查看硬盘情况，并通过该软件进行格式化操作。查看硬盘分区的UUID：
```
sudo blkid
```

在已有的目录新建挂载点`mkdir xxx`。编辑系统挂载配置文件`/etc/fstab`，按照已有格式添加新硬盘分区信息到末尾。格式为：
```
设备名称 挂载点 分区类型 挂载选项 dump选项 fsck选项
```

`dump选项`为0，就表示从不备份。如果上次用dump备份，将显示备份至今的天数。`fsck选项`为启动时的检查顺序。为0就表示不检查，`/`分区永远都是1，其它的分区只能从2开始，当数字相同就同时检查。重新启动后生效。

## NumLock
登录时启用Numlock：
```
sudo apt-get -y install numlockx
sudo sed -i 's|^exit 0.*$|# Numlock enable\n[ -x /usr/bin/numlockx ] \&\& numlockx on\n\nexit 0|' /etc/rc.local
sudo reboot
```
