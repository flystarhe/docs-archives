# Bash

## Mirrors
```
# sed -i 's/http:\/\/archive.ubuntu.com/https:\/\/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

[清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn)

## SSH
[OpenSSH Server](https://ubuntu.com/server/docs/service-openssh)远程登录设置：
```
apt-get install openssh-server vim
vim /etc/ssh/sshd_config
## ClientAliveInterval 60
## ClientAliveCountMax 3
## PermitRootLogin yes
## sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config
service ssh restart
```

很少需要手动为SSH服务器提供命令选项，`/usr/sbin/sshd -D`在不要从终端分叉并脱离终端时很有用。大多数Linux发行版都用`systemctl`启动服务。如果没有`systemctl`，请使用`service`命令。远程登录`ssh root@localhost -p 7000`。如果不希望每次输入密码，则添加公钥到`~/.ssh/authorized_keys`。如果您还没有SSH密钥，则必须[生成一个新的SSH密钥](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)。`~/.ssh/config`配置文件：
```
Host s156
  HostName 10.0.1.156
  Port 7000
  User root
  IdentityFile ~/.ssh/id_rsa
```

## Swith user

* `su user`切换用户，不切换环境变量。
* `su - user`完整的切换到新的用户环境。
* `wsl -u root`Windows子系统以`root`身份登录。

## ZSH
```
apt-get update
apt-get install -y zsh
zsh --version
echo $SHELL
chsh -s $(which zsh)
```

## Notes
`su -c "command" user`表示以`user`身份执行命令`command`，如果添加指令到`/etc/rc.local`中，则开机自启动。

计算文件校验值：Windows系统`certutil -hashfile filename sha256`，Linux系统`sha256sum filename`。

Ubuntu镜像加速：`sed -i 's/http:\/\/archive.ubuntu.com/https:\/\/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list`。

使用别名：
```
alias ll="ls -alF"
alias gs="git status"
alias gl="git log --oneline -5"
alias ga="git add ."
alias gc="git commit -m"
alias gd="git diff"
```

压缩和解压：
```
zip -r abc123.zip abc 123.txt  # abc目录和123.txt压缩为abc123.zip
unzip mydata.zip -d mydatabak  # mydata.zip解压到mydatabak
unzip -l mydata.zip  # 查看内容结构
```

`rsync`命令是一个远程数据同步工具，可通过LAN/WAN快速同步多台主机间的文件：
```
rsync -avz root@hk:/data/place_pulse_downloader/images_pulse .
```

软链接：
```
rm -rf data/coco  # 末尾没有斜杠
ln -s /real/dataset/path data/coco
```

循环/批量解压缩：
```
for t in *.zip; do echo "$(dirname ${t})/$(basename ${t})"; done
for t in *.zip; do unzip -q "${t}" -d "zip_$(basename ${t} .zip)"; done
for v in {300..303}; do t="zips/${v}.zip" && unzip -q "${t}" -d "zip_$(basename ${t} .zip)"; done
for i in {5..15}; do v=`echo ${i} | awk '{printf("%04d", $0)}'` && echo "zips/data_${v}.zip"; done
```

批量删除(Jupyter)：
```
!rm -rfv {' '.join(glob.glob(os.path.join(WORK_DIR, "epoch_*"))[:-2])}
```
