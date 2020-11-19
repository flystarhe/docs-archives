# PyCharm

## 安装
PyCharm有三个版本:专业版,社区版和教育版.在社区和教育的版本都是开放源代码的项目,它们是免费的,但他们有较少的特点.PyCharm Edu提供课程并帮助您学习Python编程.专业版是商业,并提供了一个优秀的工具集和功能.有关详细信息,请参阅版本[比较矩阵](https://www.jetbrains.com/pycharm/features/editions_comparison_matrix.html).

Ubuntu 16.04及更高版本:
```bash
sudo snap install pycharm-community --classic
```
可选`pycharm-professional`,`pycharm-educational`.终端执行`pycharm-community`,`pycharm-professional`,或`pycharm-educational`.

`snap`不可用,推荐的手动安装位置是`/opt`:
```bash
sudo tar xfz pycharm-*.tar.gz -C /opt/
cd /opt/pycharm-*/bin
pycharm.sh
```

## 首次运行
当您第一次启动PyCharm时,或者从先前版本升级后,将打开`完成安装`对话框,您可以在其中选择是否要导入IDE设置.接下来,系统将提示您选择UI主题.您可以选择Default和Darcula主题.下一步,PyCharm会提示您从PyCharm插件存储库下载并安装其他插件.

## 首个项目
完成初始PyCharm配置后,将显示`欢迎`屏幕.它允许您:打开现有项目,或从版本控制系统检出现有项目.

创建新项目时,需要配置项目解释器`Project Interpreter`:`New environment using`可选`Conda/Virtual env/Pipenv`.或者`Existing interpreter`指定已存在的解释器.

## 使用版本控制
使用`VCS | VCS Operations Popup`快速调用任何与VCS相关的命令.常用的命令有`Create Git Repository...`,`Update Project...`,`Commit...`,`Push...`.以及`VCS | Git`中的`Remotes...`.
