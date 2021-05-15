# GitHub
[GitHub.com Help Documentation](https://help.github.com/en/github)涵盖了使用Git，拉取请求，问题，Wiki，要点以及充分利用GitHub进行开发所需的一切。

* [docs.github.com](https://docs.github.com/cn)

## Searching on GitHub
[Searching on GitHub](https://help.github.com/en/github/searching-for-information-on-github/searching-on-github)可以全局搜索所有GitHub，也可搜索特定仓库或组织。在GitHub上搜索，需要先了解[搜索语法](https://help.github.com/cn/github/searching-for-information-on-github/understanding-the-search-syntax)。

通过`in`限定符，您可以将搜索限制为仓库名称`name`、仓库说明`description`、自述文件内容`readme`或这些的任意组合。如果省略此限定符，则只搜索仓库名称和说明。

* `jquery in:name`匹配其名称中含有`jquery`的仓库。
* `jquery in:name,description`匹配其名称或说明中含有`jquery`的仓库。
* `stars:>=500 fork:true language:php`匹配具有至少500星，包括复刻且以PHP编写的仓库。
* `webos created:<2011-02-01`匹配具有`webos`字样，且在2011年1月之前创建的仓库。
* `css pushed:>2013-02-01`匹配具有`css`字样，且在2013年1月之后收到推送的仓库。

更多内容请参阅[url](https://help.github.com/cn/github/searching-for-information-on-github/searching-for-repositories)。

## Large File Storage
Git仓库包含每个文件的每个版本。但对于一些文件类型来说，这是不实际的。多次修订大文件会增加仓库其他用户克隆和获取的时间。GitHub限制了存储库中允许的文件大小，如果文件大于最大文件限制，它将阻止推送到存储库。如果尝试添加或更新大于50MB的文件，则会收到来自Git的警告。所做的更改仍将成功推送到您的存储库，但是您可以考虑删除提交以最大程度地降低性能影响。有关更多信息，请参阅[从存储库的历史记录中删除文件](https://help.github.com/cn/github/managing-large-files/removing-files-from-a-repositorys-history)。GitHub会阻止超过100MB的推送。

* 分发大型二进制文件

  除了分发源代码外，一些项目还需要分发大型文件，例如二进制文件或安装程序。如果需要在存储库中分发大文件，则可以在GitHub上创建发行版`releases`。通过发行版，您可以打包软件，发行说明以及指向二进制文件的链接，以供其他人使用。如果您定期将大文件推送到GitHub，请考虑使用Git大文件存储(Git LFS)。更多信息请参阅[大型文件版本管理](https://help.github.com/cn/articles/versioning-large-files)。

## Removing files from a repository's history
要从仓库中删除大文件，必须将其从本地仓库和GitHub中完全删除。

* 删除在最近未推送的提交中添加的文件

  如果文件使用最近的提交添加，而您尚未推送到GitHub，您可以删除文件并修改提交：
  ```
  git rm --cached giant_file
  git commit --amend -CHEAD
  git push
  ```

* 删除先前提交中添加的文件，从仓库的历史记录中清除文件

  要从仓库的历史记录中彻底删除不需要的文件，您可以使用`git filter-branch`命令。为说明其工作方式，我们将向您展示如何从仓库的历史记录中删除文件，然后将其添加到`.gitignore`以确保不会意外重新提交。
  ```
  git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
  cd YOUR-REPOSITORY
  git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch PATH-TO-YOUR-FILE-WITH-SENSITIVE-DATA" \
  --prune-empty --tag-name-filter cat -- --all
  echo "YOUR-FILE-WITH-SENSITIVE-DATA" >> .gitignore
  git add .gitignore
  git commit -m "Add YOUR-FILE-WITH-SENSITIVE-DATA to .gitignore"
  git push origin --force --all
  ```

## working tree, index file, commit
`working tree`就是你所工作在的目录，每当你在代码中进行了修改，`working tree`的状态就改变了。`index file`是索引文件，每当我们使用`git add`命令后，`index file`的内容就改变了。`commit`是最后的阶段，只有`commit`了，代码才真正进入了仓库。理解`git-diff`：

* `git diff`查看`working tree`与`index file`的差别的。
* `git diff HEAD`查看`working tree`和`commit`的差别的。
* `git diff --cached`查看`index file`与`commit`的差别的。

## Git Notes

* [3.2 Git 分支 - 分支的新建与合并](https://git-scm.com/book/zh/v2)
* [3.3 Git 分支 - 分支管理](https://git-scm.com/book/zh/v2)
* [3.4 Git 分支 - 分支开发工作流](https://git-scm.com/book/zh/v2)
