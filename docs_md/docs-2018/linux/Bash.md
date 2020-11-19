# Bash
每个职业都有最常用的工具.对于许多系统管理员来说,shell可能是比较熟悉的.在大多数Linux和其他类Unix系统上,默认的shell是Bash.

## rm
Bash shell支持丰富的文件模式匹配符,例如:

- `*`:匹配所有文件
- `?`:匹配文件名中的单个字母
- `[..]`:匹配括号中的任意一个字母

例如:`rm -rf [0-9]*`删除所有以数字开头的文件.

### 扩展模式
用系统内置的`shopt -s extglob`命令开启shell中的`extglob`选项,然后就可以使用扩展模式:

- `?`:匹配零次或一次给定的模式
- `*`:匹配零次或多次给定的模式
- `+`:至少匹配一次给定的模式
- `@`:匹配一次给定的模式
- `!`:不匹配给定模式

关闭`extglob`选项方法为`shopt -u extglob`.示例:
```bash
## 仅保留 file1 文件
rm !(file1)

## 仅保留 file1 和 file2 文件
rm !(file1|file2)

## 仅保留特定后缀的文件
rm !(*.zip|*.iso)

## 仅保留特定前缀的文件
rm -rf !(latest*|loss*|opt*)
```

## 使用别名
为`mv`和`rm`等命令设置别名,指向`mv -i`和`rm -i`.这将确保运行`rm -f /boot`至少要求你确认.在红帽企业版Linux中,如果你使用`root`帐户,则默认设置这些别名.如果你还要为普通用户帐户设置这些别名,只需将这两行放入主目录中名为`.bashrc`的文件中(这些也适用于`sudo`):
```bash
alias mv='mv -i'
alias rm='rm -i'
alias c='clear'
alias l='ls -CF'
alias la='ls -A'
alias ll='ls -alF'
alias ls='ls --color=auto'
```

执行`alias`命令会显示一个所有已定义别名的列表.若要删除已经设置的别名,使用内建命令`unalias`,`unalias -a`表示删除所有已设置的别名,`unalias alias-name`表示仅删除`alias-name`.

## 比较文件
你可以使用`diff`来一行行第比较文件,而一个名为`colordiff`的工具可以为`diff`输出着色:
```bash
colordiff -c pix2pix_hej_model.py pix2pix_model.py
```

## ssh
检查现有的SSH密钥,终端输入`ls -al ~/.ssh`.[或生成一个新的SSH密钥用于身份验证](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh):
```
# 替换您的GitHub电子邮件地址
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 在后台启动ssh-agent
eval "$(ssh-agent -s)"
# 默认文件位置为
ssh-add ~/.ssh/id_rsa
# 测试SSH连接
ssh -T git@github.com
```

不要创建`ssh`别名,代之以`~/.ssh/config`配置文件.它的选项更加丰富,例子:
```
Host s156
  Hostname 10.0.1.156
  IdentityFile ~/.ssh/id_rsa
  User root
```

然后,添加`~/.ssh/id_rsa.pub`到目标主机的`~/.ssh/authorized_keys`.若执行`ssh s156`时,报错`Bad owner or permissions on /home/hejian/.ssh/config`,则在`~/.ssh`目录执行`chmod 700 *`.

## 压缩和解压
```
zip -r abc123.zip abc 123.txt  #abc目录和123.txt压缩为abc123.zip
unzip mydata.zip -d mydatabak //mydata.zip解压到mydatabak
unzip -l mydata.zip //查看内容结构
```

## 合并和切分
```
# 合并1.txt和2.txt文件内容到all.txt
cat 1.txt 2.txt >> all.txt
# 每100行数据为一个新文本存到filedir目录
split -l 100 -d trainfile.txt filedir/
```

## 远程数据同步
rsync命令是一个远程数据同步工具,可通过LAN/WAN快速同步多台主机间的文件:
```
rsync -avz root@hk:/data/place_pulse_downloader/images_pulse .
```

## 从终端打开当前目录
```
nautilus .
```

## 软链接
```
rm -rf data/coco  # 末尾没有斜杠
ln -s $DATA_ROOT data/coco
```
