# Git

## CentOS 7
RHEL和衍生产品通常会发布较旧版本的git。您可以下载tarball并从源代码进行构建，也可以使用第三方存储库（例如[IUS](https://ius.io/setup)社区项目）来获取git的最新版本。
```
yum install \
https://repo.ius.io/ius-release-el7.rpm \
https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

yum remove git
yum install git222
git --version
```

自动保存账号密码：
```
git config --global credential.helper store
```

进行一次`push`或`pull`，输入一次密码以后就不用了。`~/.gitconfig`内容如下：
```
[user]
        email = flystarhe@qq.com
        name = hejian
[credential]
        helper = store
```

密码等敏感信息被记录在`~/.git-credentials`中：
```
https://username:password@github.com
```

## .gitignore
```
# https://github.com/github/gitignore
tmp
.vscode
*.py[cod]
__pycache__

# Mac
.DS_Store

# Linux
.Trash-*

# Windows
Thumbs.db
ehthumbs.db
[Dd]esktop.ini

# Jupyter Notebook
.ipynb_checkpoints

# mkdocs documentation
/site

# Sphinx documentation
docs/_build/

# Data directory
data/*
!data/readme.md

# Outputs
results

# Git
!.gitkeep
```

## Git服务
[Git服务器 - 私有仓库](https://flystarhe.github.io/docs-2018/linux/Git/#git_1)

## Notes
```
# 停止跟踪当前跟踪的文件
git rm --cached

# 修改最后一次提交
git commit --amend -CHEAD
git commit --amend -m <message>

# 回滚到指定的提交
git reset --hard xxxxxxx

# 远程仓库
git remote add origin https://github.com/user/proj.git
git remote set-url origin https://github.com/user/proj.git
git remote rename <old> <new>
git remote remove <name>

# 分支管理
git branch
git branch -v            # 查看各分支最新提交
git branch testing       # 创建`testing`分支
git checkout testing     # 切换`testing`分支
git branch -d testing    # 删除`testing`分支
git checkout -b testing  # 新建分支并切换到新分支

# 远程分支
git branch -vv
git clone -b main https://github.com/user/proj.git
git clone --recurse-submodules --depth 1 https://github.com/user/proj.git
git fetch <远程版本库名> <分支名>
git pull <远程版本库名> <远程分支名>:<本地分支名>
git push <远程版本库名> <本地分支名>:<远程分支名>
git branch -u origin/branch-name [<local-branch>]

# 分支比较
git diff origin/main other/main --stat  # 显⽰分⽀间差异的部分
git diff origin/main other/main         # 显⽰分⽀间所有差异⽂件的详细差异

# 标签管理
git tag
git tag -l 'v1.4*'
git tag -a v1.4 -m "my version 1.4"  # 附注标签
git tag v1.4-lw                      # 轻量标签
git tag -a v1.4 xxxxxxx              # 追加标签
git tag -d v1.4-lw                   # 移除本地标签
git push origin :refs/tags/v1.4-lw   # 移除远程标签
git checkout -b v1.4 v1.4rc1         # tag: v1.4rc1
git checkout -B main v1.4rc1         # tag: v1.4rc1
git fetch origin v1.4rc1:tmp         # tag: v1.4rc1

# 日志管理
git log --oneline -5

# 清除历史
git checkout --orphan latest
git add .
git commit -m "v1.0"
git branch -M main
git push -f origin main
git branch -u origin/main main

# 子模块
git submodule add https://github.com/user/proj.git module_name
git submodule init
git submodule update
git rm --cached module_name
rm -rf module_name

# dev1
git fetch https://github.com/user/proj.git main:dev
git checkout dev
...work...
git push origin dev:dev
git branch -d dev

# dev2
git remote add other https://github.com/user/proj.git
git remote show other
git fetch other main
git branch -a
git checkout -b repo-other other/main
...work...
git checkout main
git merge repo-other
git branch -d repo-other

# dev3
git remote add other https://github.com/user/proj.git
git remote show other
git fetch other main
git branch -a
git merge other/main
```

## 参考链接：
* [Git Book](https://git-scm.com/book/zh/v2)
* [Git Guide](https://github.com/git-guides/)
* [Git 分支 - 分支的新建与合并](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E7%9A%84%E6%96%B0%E5%BB%BA%E4%B8%8E%E5%90%88%E5%B9%B6)
* [Git 分支 - 分支开发工作流](https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%88%86%E6%94%AF%E5%BC%80%E5%8F%91%E5%B7%A5%E4%BD%9C%E6%B5%81)
