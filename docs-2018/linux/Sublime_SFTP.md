# Sublime SFTP
Sublime有个叫`SFTP`的插件,可以通过它直接打开远程机器上的文件进行编辑,并在保存后直接同步到远程linux服务器上.用`Package Control`安装`SFTP`插件.

## 配置
`File->SFTP/FTP->Setup Server`打开配置文件,调整以下参数即可:

- `host`: 你的服务器地址
- `user`: 登录服务器的用户名
- `password`: 登录服务器的密码
- `remote_path`: 想要远程操作的目录

这种方法要求服务器可以通过`sftp`或`ftp`连接上去,也就是服务器上需要运行有类似`ftp server`的东西.

`ftp server`和`sublime`都配置好后便可通过`File->SFTP/FTP/Browse Server`来查看服务器上的目录和文件了,然后可根据提供的命令重命名目录,编辑文件等.编辑好的文件保存后可立即同步到服务器.

如果你想将服务器上的目录拉倒sublime里面,就如同打开本地的文件一样,操作如下:

1. 先在本地创建一个文件夹,用sublime打开
2. 右键目录名,选择`SFTP/FTP->Map to Remote`
3. 弹出配置文件,在配置文件中修改相关配置项的参数
4. 右键目录名,`SFTP/FTP->Download Folder`,然后等待同步完成

完成你的操作后,你可以通过右键目录名,点击`SFTP/FTP->Upload Folder`,即可同步到服务器.