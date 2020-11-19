title: Install Ubuntu 16.04
date: 2017-07-03
tags: [Linux,Ubuntu]
---
本文记录了博主[Ubuntu 16.04](http://releases.ubuntu.com/)安装体验的关键过程，从准备磁盘空间、制作启动盘、开始安装、检查更新、安装驱动等内容。

<!--more-->
## 准备磁盘空间
`此电脑`右键`管理`，导航到`存储 -> 磁盘管理`，选择不用盘符`右键 -> 删除卷`，或选择较大的盘符`右键 -> 压缩卷`。有两颗硬盘的豪可以跳过。

## 制作启动盘
下载最新版的`UltraISO`，即使是试用也没关系。首先，点击`文件 -> 打开`，选择你的ISO文件；然后，点击`启动 -> 写入磁盘映像`，确定`硬盘驱动器`正确选择，写入方式为`USB-HDD+`，单击`写入`开始。（建议先格式化）

## 开始安装
重启电脑，进Boot界面，设置USB启动优先，进入引导界面，按照提示操作就好了。建议安装英文的，除非你准备受那些中文目录的折磨。常见的分区方式：

1. `/`文件系统和`SWAP`分区
2. `/`文件系统和`SWAP`分区，加`/home`文件系统
3. 再详细点，再多个`/boot`文件系统、`/var`文件系统等
4. 不属于Linux目录树的`/back`（名字自己定，这是自己用来存放备份数据的地方）

100M，主分区，选择“保留 BIOS 启动区域/不使用此分区”；40960M，主分区，挂载点`/`；4096M，主分区，选择“交换空间”；剩余空间，主分区，挂载点`/home`。安装启动引导器的设备选刚才的“保留 BIOS 启动区域/不使用此分区”。

### 安装黑屏解决方法(1)
`Ubuntu 16.04 + GTX1070`，安装出现黑屏，是`Ubuntu 16.04`集成的驱动不支持`GTX1070`所致。启用集成显卡就能解决，虽然`Ubuntu`不支持`GTX1070`的开箱即用，但支持集成显卡的开箱即用却没有任何问题。进入Bios设置，进入`Advanced\Chipset Configuration`，修改`IGPU Multi-Monitor`值`Disabled`为`Enabled`，保存并退出。

注意,部分主板可能是,通过BIOS更改初始显卡选项`Graphics Configuration/Primary Display`,设置集显(IGFX)或独显(PCIE).

### 安装黑屏解决方法(2)
此方法是网上搜索的，完全就是个坑，强烈不推荐。在引导界面(就是Bios画面之后显示的画面)，按`E`到达选择语言界面，选择`中文简体`，按上下键选中`安装 Ubuntu`，按`F6`设置引导选项，替换`splash`后面的`---`为`nomodeset`，`Esc`离开图形引导菜单并启动文本模式界面。
```
Boot Options ed boot=..initrd=/casper/initrd.lz quiet splash ---
  =>
Boot Options ed boot=..initrd=/casper/initrd.lz quiet splash nomodeset
```

### 启动黑屏解决方法
开机，进入`grub`画面(如果没有别的OS，开机时按住`Shift`不放)，按`E`编辑`Ubuntu`的启动选项，找到`no splash`，在后面添加`nomodeset`，`F10`保存并启动，即可进入系统。

立即显卡驱动者，请跳过。进入系统之后，按`Ctrl+Alt+F1`切到命令行模式，打开`/etc/default/grub`，修改如下：
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
  =>
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nomodeset"
```

执行`sudo update-grub`更新`grub`配置信息。参考：[Ubuntu16.04开机启动字符界面](http://jingyan.baidu.com/article/948f5924ee2a5dd80ff5f9e4.html)

## 检查更新
要获取新`Ubuntu 16.04 LTS`更新可以在`Unity Dash`打开`软件更新器 > 检查更新`按钮进行更新。

## 安装驱动
`Ubuntu 16.04`支持大多数`Nvidia`和`Intel`显示硬件的开箱即用，当然你也可以安装免费的开源驱动或其它闭源驱动。要安装显卡驱动你可以在`软件和更新 > 附加驱动`选项卡中进行选择。对于使用`Nvidia`的游戏玩家，可以[下载官方驱动程序](http://www.geforce.cn/drivers)。图省事可以使用`sudo apt-get install nvidia-375`命令，推荐方法。

`Ubuntu`系统集成的显卡驱动程序是`nouveau`，它是第三方为`NVIDIA`开发的开源驱动，我们需要先将其屏蔽才能安装`NVIDIA`官方驱动。`sudo vim /etc/modprobe.d/blacklist.conf`添加黑名单：
```
blacklist nouveau
```

执行`sudo update-initramfs -u`更新，修改后需要重启系统。确认`Nouveau`已经被你干掉，使用命令`lsmod | grep nouveau`。

按`Ctrl+Alt+F1`切到命令行模式，关闭图形环境：
```
sudo service lightdm stop
```

先删除旧的驱动(完全卸载安装包，包括删除配置文件)：
```
sudo apt-get purge nvidia*
```

安装驱动程序[NVIDIA-Linux-x86_64-375.66.run](http://www.geforce.cn/drivers)：
```
sudo sh NVIDIA-Linux-x86_64-375.66.run
```

重新启动图形环境：
```
sudo service lightdm start
```

按`Ctrl+Alt+F7`切到图形界面。用`nvidia-smi`命令查看显卡和驱动情况，列出GPU的信息列表则表示驱动安装成功。如果安装后驱动程序工作不正常，使用下面的命令进行卸载：
```
sudo sh NVIDIA-Linux-x86_64-375.66.run --uninstall
```

这时,你可能切到图形界面,在登录界面耗着,死活进不了图形界面.可能原因是你卸载了所有图形驱动,仅安装了独显驱动,但是你的主板的显卡初始选项确是集显(IGFX).所以,你需要重启,修改BIOS初始显卡选项为独显(PCIE).问题解决.

## Unity位置
从`Ubuntu 16.04`开始，用户已经可以手动选择将`Unity栏`放在桌面左侧或是底部显示。移动到桌面底部/左部：
```
gsettings set com.canonical.Unity.Launcher launcher-position Bottom/Left
```

## 点击图标最小化
`Ubuntu 16.04 LTS`也支持了点击应用程序`Launcher图标`即可`最小化`的功能，不过还是需要用户进行手动启用。

方法有两种，你可以[安装`Unity Tweak Tool`](http://jingyan.baidu.com/article/d45ad148bc51bb69552b80dc.html)图形界面工具之后在`Unity > Launcher > Minimise`中进行配置，或直接在终端中使用如下命令启用：
```
gsettings set org.compiz.unityshell:/org/compiz/profiles/unity/plugins/unityshell/ launcher-minimize-window true
```

## 搜狗输入法
首先下载搜狗拼音[sogoupinyin_2.1.0.0086_amd64.deb](http://pinyin.sogou.com/linux/)，查看帮助转[help](http://pinyin.sogou.com/linux/help.php)。

## sublime text
通过[Linux repos]安装:
```
sudo apt-get install apt-transport-https
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
```

有时你希望,或被迫使用[?.tar.bz2](https://www.sublimetext.com/3)安装:
```
wget https://download.sublimetext.com/sublime_text_3_build_3143_x64.tar.bz2
tar -jxf sublime_text_3_build_3143_x64.tar.bz2 -C /opt
/opt/sublime_text_3/sublime_text
```

这样通过命令运行,太麻烦.所以,执行`cp /opt/sublime_text_3/sublime_text.desktop /usr/share/applications`.

### 解决中文输入问题
`cd ~ && sudo vim sublime-imfix.c`:
```
#include <gtk/gtkimcontext.h>
void gtk_im_context_set_client_window (GtkIMContext *context, GdkWindow *window)
{
  GtkIMContextClass *klass;
  g_return_if_fail (GTK_IS_IM_CONTEXT (context));
  klass = GTK_IM_CONTEXT_GET_CLASS (context);
  if (klass->set_client_window)
    klass->set_client_window (context, window);
  g_object_set_data (G_OBJECT(context), "window", window);
  if (!GDK_IS_WINDOW (window))
    return;
  int width = gdk_window_get_width (window);
  int height = gdk_window_get_height (window);
  if (width != 0 && height !=0)
    gtk_im_context_focus_in (context);
}
```

安装编译环境:
```
sudo apt-get install build-essential
sudo apt-get install libgtk2.0-dev
```

编译成共享库`libsublime-imfix.so`:
```
gcc -shared -o libsublime-imfix.so sublime-imfix.c `pkg-config --libs --cflags gtk+-2.0` -fPIC
```

移动共享库到`/opt/sublime_text`目录下:
```
sudo mv libsublime-imfix.so /opt/sublime_text
```

`sudo vim /usr/bin/subl`:
```
#!/bin/sh
export LD_PRELOAD=/opt/sublime_text/libsublime-imfix.so
exec /opt/sublime_text/sublime_text "$@"
```

以后通过终端执行`subl`启动sublime就可以输入中文了.

## 笔记

### 命令行和图形界面切换
从图形界面切换到命令行模式，我们可以通过按`CTRL+ALT+F1..F6`，为什么是`F1..F6`，因为在Linux中一般有F1到F6多个命令行字符终端，也就是说我们可以同时打开最多6个命令行界面。从命令行模式切换回图形界面，我们可以通过按`CTRL+ALT+F7`。

### 内存使用情况
```
free -m
```

### CPU使用情况
```
top -i
```

使用工具:
```
sudo apt-get install htop
htop
```

### GPU使用情况
```
nvidia-smi
```

配合watch,每隔1s刷新:
```
watch -n 1 nvidia-smi
```

### 防火墙
```bash
ufw status
ufw enable
ufw default deny
ufw allow 9000
ufw delete allow 80
ufw allow from 192.168.1.1
ufw reload
ufw disable
```
