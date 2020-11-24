# Git

## 基础

### create a new repository
```
$ echo "# ?" >> readme.md
$ git init
$ git add readme.md
$ git commit -m "first commit"
$ git remote add origin https://github.com/user/proj_name.git
$ git push -u origin master
```

### push an existing repository
```
$ git remote set-url origin https://github.com/user/proj_name.git
$ git push -u origin master
```

### remote
```
$ git remote rename <old> <new>
$ git remote remove <name>
```

### .gitignore
```
!/docs/**/*
/.*/
/**/tmps/
/**/.cache/
/**/__pycache__/
/**/.ipynb_checkpoints/
/**/.DS_Store
/**/*.pyc
/**/*.bak
/docs_md/**/*.html
```

### clean
```
git checkout --orphan latest
git add .
git commit -m "v1.0"
git branch -D master
git branch -m master
git push -f origin master
git branch -u origin/master master
```

### dev1
```
git fetch origin master:dev
git checkout dev
...work...
git push origin dev:dev
git branch -d dev
```

>如果存在冲突则需要先解决冲突，拉取`git fetch origin dev`，合并`git merge origin/dev`，编辑冲突内容再提交。

### dev2
```
git fetch https://github.com/user/proj_name.git master:dev
git checkout dev
...work...
git push https://github.com/user/proj_name.git dev:dev
git branch -d dev
```

### dev3
```
# 1 添加待合并的远程仓库并查看分支
git remote add other https://github.com/user/proj_name.git
git remote show other
# 2 从远程仓库中抓取master分支
git fetch other master
git branch -a
# 3 将远程仓库的master分支作为新分支checkout到本仓库
git checkout -b repo-other other/master
git branch -a
# 4 切换会本仓库的master分支
git checkout master
git branch -a
# 5 合并到本仓库的master分支
git merge repo-other
# 6 删除`repo-other`分支
git branch -d repo-other
```

上述方法虽然细致,但很多时候是这样:
```
$ git merge other/master
```

## 打标签
像其他版本控制系统一样，Git可以给历史中的某一个提交打上标签，以示重要。比较有代表性的是人们会使用这个功能来标记发布结点（v1.0等等）。

### 列出标签
```
git tag
git tag -l 'v1.8.5*'
```

### 创建标签
Git使用两种主要类型的标签：轻量标签与附注标签。一个轻量标签很像一个不会改变的分支，它只是一个特定提交的引用。然而，附注标签是存储在Git数据库中的一个完整对象。通常建议创建附注标签，但是如果你只是想用一个临时的标签，或者因为某些原因不想要保存那些信息，轻量标签也是可用的。
```
# 附注标签
git tag -a v1.4 -m "my version 1.4"
# 轻量标签
git tag v1.4-lw
```

>通过使用`git show`命令可以看到标签信息与对应的提交信息。

你也可以对过去的提交打标签，假设在`v1.2`时你忘记给项目打标签：
```
git log --pretty=oneline -5
git tag -a v1.2 9fceb02
```

默认情况下，`git push`命令并不会传送标签到远程仓库服务器上。在创建完标签后你必须显式地推送标签到共享服务器上。这个过程就像共享远程分支一样`git push origin [tagname]`。如果想要一次性推送很多标签，也可以使用带有`--tags`选项的`git push origin --tags`命令。这将会把所有不在远程仓库服务器上的标签全部传送到那里。

### 检出标签
如果你想查看某个标签所指向的文件版本，可以使用`git checkout`命令，虽然说这会使你的仓库处于“分离头指针（detacthed HEAD）”状态，这个状态有些不好的副作用。在“分离头指针”状态下，如果你做了某些更改然后提交它们，标签不会发生变化，但你的新提交将不属于任何分支，并且将无法访问，除非确切的提交哈希。因此，如果你需要进行更改，比如说你正在修复旧版本的错误，这通常需要创建一个新分支：
```
# tag: v1.0rc1
git checkout -b v1 v1.0rc1
git checkout -B master v1.0rc1

git fetch origin v1.0rc1:tmp
git checkout tmp
git branch -d master
git branch -m master
```

### 删除标签
```
git tag -d v1.4-lw
# 从远程仓库中移除这个标签
git push origin :refs/tags/v1.4-lw
```

## 分支管理
`git branch`命令不加任何参数运行,会得到当前所有分支的一个列表:
```
$ git branch -a
* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/master
```

注意`master`分支前的`*`字符,它代表现在检出的那一个分支,也就是说当前`HEAD`指针所指向的分支.如果需要查看每一个分支的最后一次提交,可以运行`git branch -v`命令.`--merged`与`--no-merged`这两个有用的选项可以过滤这个列表中已经合并或尚未合并到当前分支的分支.如果要查看哪些分支已经合并到当前分支,可以运行`git branch --merged`.查看未合并的分支,可以运行`git branch --no-merged`.

```
$ git branch testing  # 创建`testing`分支
$ git checkout testing  # 切换`testing`分支
$ git branch -d testing  # 删除`testing`分支
$ git checkout -b testing  # 新建分支并切换到新分支
```

分支比较:
```
$ git diff origin/master other/master --stat  # 显⽰分⽀间差异的部分
$ git diff origin/master other/master  # 显⽰分⽀间所有差异⽂件的详细差异
$ git log branch1 ^branch2  # 查看分⽀1中有，而分⽀2中没有的log
$ git log branch1..branch2  # 查看分⽀2中⽐分⽀1中多提交了那些内容
$ git log branch1...branch2  # 不知道谁提交的多谁提交的少，单纯想知道有什么不⼀样
$ git cherry -v origin/master testing  # 比较本地testing分支和远程master分支的差别
$ git cherry -v origin/master  # 比较本地HEAD分支和远程master分支的差别
$ git cherry -v master  # 比较本地HEAD分支和本地master分支的差别
```

- `git diff`查看尚未暂存的文件更新了哪些部分
- `git diff HEAD`查看未暂存文件与最新提交文件快照的区别
- `git diff --cached`查看已暂存文件和上次提交时的快照之间的差异
- `git diff <index1> <index2>`查看不同快照之间的区别
- `git reset --hard xxx`回滚到指定版本
- `git log --oneline -5`显示最近提交

## 日志管理
- `git log -p -2`显示每次提交的内容差异
- `git log --stat -2`显示每次提交的统计信息
- `git log --oneline -5`将每个提交放在一行显示
- `git log --pretty=oneline -5`还有`short/full/fuller`
- `git log --simplify-by-decoration`选择某个分支或标签引用的提交
- `git log --simplify-by-decoration --oneline --graph`查看分支图

## 远程分支
查看分支跟踪关系:
```
$ git branch -vv
```

下载指定分支命令为:
```
$ git clone -b master https://github.com/user/proj_name.git
```

把远程版本库取回本地:
```
$ git fetch <远程版本库名> <分支名>
```

取回远程版本库某个分支的更新,并与本地的指定分支合并:
```
$ git pull <远程版本库名> <远程分支名>:<本地分支名>
```

>与当前分支合并,则冒号后面的部分可以省略.

创建本地分支和远程分支的链接关系:
```
$ git branch -u origin/branch-name [<local-branch>]
$ git branch --set-upstream-to=origin/branch-name [<local-branch>]
$ git branch --unset-upstream [<local-branch>]
```

如果远程分支不存在则使用以下方式:
```
$ git push -u origin local-branch:remote-branch
```

将本地分支的更新推送到远程版本库,并与远程版本库指定分支合并:
```
$ git push <远程版本库名> <本地分支名>:<远程分支名>
```

>若省略本地分支名,则表示删除指定的远程分支,如`git push origin :master`,等价`git push origin --delete master`.

## 子模块

### add
通过在`git submodule add`命令后面加上想要跟踪的项目URL来添加新的子模块。
```
$ git submodule add https://github.com/user/proj_name.git module_proj_name
```

### clone
克隆一个含有子模块的项目时，默认会包含该子模块目录，但其中还没有任何文件。你必须运行两个命令，`git submodule init`用来初始化本地配置文件，而`git submodule update`则从该项目中抓取所有数据并检出父项目中列出的合适的提交。
```
$ git clone https://github.com/user/proj_name.git && cd proj_name
$ git submodule init
$ git submodule update
```

不过还有更简单一点的方式。如果给`git clone`命令传递`--recurse-submodules`选项，它就会自动初始化并更新仓库中的每一个子模块。
```
$ git clone --recurse-submodules --depth 1 https://github.com/user/proj_name.git
```

### update
运行`git submodule update --remote`，Git将会进入子模块然后抓取并更新。Git默认会尝试更新所有子模块，所以如果有很多子模块的话，你可以传递想要更新的子模块的名字。

### delete
如果需要移除子模块,请如下操作:
```
$ git rm --cached module_proj_name
$ rm -rf module_proj_name
```

## Git服务器
GitHub就是一个免费托管开源代码的远程仓库.但是对于某些视源代码如生命的商业公司来说,既不想公开源代码,又舍不得给GitHub交保护费,那就只能自己搭建一台Git服务器作为私有仓库使用.

安装`git`:
```
$ apt-get install git
```

添加`git`用户:
```
$ useradd git -m -s /bin/bash
$ su - git
```

创建证书登录:收集所有需要登录的用户的公钥,就是他们自己的`id_rsa.pub`文件,把所有公钥导入到`/home/git/.ssh/authorized_keys`文件里,一行一个.

>请注意设置`sudo chmod -R 750 .ssh/`,否则会失败.

初始化仓库.假定是`/home/git/sample.git`,在`/home/git`目录下输入命令:
```
$ git init --bare sample.git
```

Git会创建一个裸仓库,裸仓库没有工作区,因为服务器上的Git仓库纯粹是为了共享,并且服务器上的Git仓库通常都以`.git`结尾.

禁用shell登录:
```
git:x:1001:1001::/home/git:/bin/bash
改为:
git:x:1001:1001::/home/git:/usr/bin/git-shell
```

克隆远程仓库(客户机):
```
$ git clone git@server:/home/git/sample.git
```

## 参考资料:
- [Git Book](https://git-scm.com/book/zh)