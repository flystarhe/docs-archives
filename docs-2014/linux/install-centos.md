title: Install CentOS 7
date: 2017-07-03
tags: [Linux,CentOS]
---
本文记录了博主`CentOS 7`安装体验的关键过程，从准备磁盘空间、制作启动盘、开始安装到引导项修复均有涉猎。参考[安装Windows 10 + Centos 7双系统共存](http://xueliang.org/article/detail/20160912181035032)。

<!--more-->
## 准备磁盘空间
`此电脑`右键`管理`，导航到`存储 -> 磁盘管理`，选择不用盘符`右键 -> 删除卷`，或选择较大的盘符`右键 -> 压缩卷`。有两颗硬盘的豪可以跳过。

## 制作启动盘
下载最新版的`UltraISO`，即使是试用也没关系。首先，点击`文件 -> 打开`，选择你的ISO文件；然后，点击`启动 -> 写入磁盘映像`，确定`硬盘驱动器`正确选择，写入方式为`USB-HDD+`，单击`写入`开始。（建议先格式化）

## 开始安装
重启电脑，进Boot界面，设置USB启动优先。很快就能看到有`Install CentOS 7`字样的画面，按`Tab`键，屏幕下方的倒计时变为`> vmlinuz initrd=initrd.img inst.stage2=hd:LABEL=CentOS\x207\x20x86_64 rd.live.check quiet`，修改为`vmlinuz initrd=initrd.img linux dd quiet`，然后回车。看到由`DEVICE`、`TYPE`、`LABEL`和`UUID`组成的表格，`LABEL`显示驱动器名称，据此找到你的U盘。强制关闭计算机后再开机，回到刚才倒计时那个界面，依旧按`Tab`键，修改启动参数：
```
> vmlinuz initrd=initrd.img inst.stage2=hd:/dev/sdb1 quiet
```

`sdb1`换成你U盘对应的值，于是出现了大堆的字符串和OK，不用理会，等着就好。不久之后，就能看到`CENTOS 7 安装`图形界面。

## 引导项
对于`Windows 10 + CentOS 7`双系统，安装好`CentOS 7`后，重启找不到`Windows 10`启动项。

### 找回Windows 10启动项(1)
打开终端，安装`ntfs-3g`：
```
yum install -y ntfs-3g
```

更新`Grub2`启动菜单，找回`Windows 10`：
```
grub2-mkconfig -o /boot/grub2/grub.cfg
```

### 找回Windows 10启动项(2)
执行`vim /boot/grub2/grub.cfg`，在第一个`menuentry`前面，添加：
```
menuentry 'Windows 10' {
    set root=(hd0,1)
    chainloader +1
}
```

重启系统。此时虽然`Windows 10`是第一个选项，但默认进入的依然是`CentOS 7`。

### 找回Windows 10启动项(3)
重启电脑发现`Windows系统引导项`不见了，只有`CentOS选项`。进入`Windows PE/Windows To Go`，在里面找到`Windows引导恢复`，点击`自动修复`。

当引导恢复成功后，你会发现`CentOS引导项`丢失了。登录Windows系统，使用EasyBCD软件增加`CentOS引导项`。CentOS系统使用`GRUB 2`引导方式，别选错了。

## 工具
- Ext2IFS：能够在Windows上读写LINUX的EXT2/3文件系统
- EasyBCD：用于设置多系统引导的一个非常好用的用具

## 笔记

### 命令行和图形界面切换
图形模式下会占用大量的系统资源，尤其占用内存。作为服务器，安装调试完毕后，应该让系统运行在文本模式(命令行)下。不管是在登录场景还是桌面场景，从图形模式切换到文本模式的快捷键是`CTRL+ALT+Fn(n=2,3,4,5)`，从文本模式切换到图形模式，按`CTRL+ALT+F1`。