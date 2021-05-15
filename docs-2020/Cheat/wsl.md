# WSL
适用于Linux的Windows子系统可让开发人员按原样运行GNU/Linux环境，包括大多数命令行工具、实用工具和应用程序，且不会产生传统虚拟机或双启动设置开销。

## 安装WSL
必须先启用“适用于Linux的Windows子系统”可选功能，然后才能在Windows上安装Linux分发版。以管理员身份打开PowerShell并运行：
```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

若要仅安装WSL1，现在应重启计算机并继续安装所选的Linux分发版，否则请等待重启并继续更新到WSL2。若要更新到WSL2，必须已更新到版本2004的内部版本19041或更高版本。`Windows + R`键入`winver`检查你的Windows版本。安装WSL2之前，必须启用“虚拟机平台”可选功能。以管理员身份打开PowerShell并运行：
```
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

重新启动计算机，以完成WSL安装并更新到WSL2。
```
wsl --set-default-version 2
```

[比较 WSL 1 和 WSL 2](https://docs.microsoft.com/zh-cn/windows/wsl/compare-versions)中可以看出，除了跨操作系统文件系统的性能外，WSL 2 体系结构在多个方面都比 WSL 1 更具优势。可以使用 Windows 应用和工具（如文件资源管理器）访问 Linux 根文件系统。尝试打开 Linux 分发版（如 Ubuntu），通过输入以下命令确保你位于 Linux 主目录中：`cd ~`。然后通过输入`explorer.exe .`（不要忘记尾部的句点），在文件资源管理器中打开 Linux 文件系统。

## 安装Linux分发版
打开 [Microsoft Store](https://aka.ms/wslstore)，并选择你偏好的Linux分发版。首次启动新安装的Linux分发版时，将打开一个控制台窗口，系统会要求你等待一分钟或两分钟，以便文件解压缩并存储到电脑上。未来的所有启动时间应不到一秒。然后，需要为新的Linux分发版创建用户帐户和密码。`wsl -u root`Windows子系统以`root`身份登录。

## 安装Windows终端
可以从 [Microsoft Store](https://aka.ms/terminal) 安装 Windows 终端。如果你无法访问，[GitHub 发布页](https://github.com/microsoft/terminal/releases)上发布有内部版本。如果从 GitHub 安装，终端将不会自动更新为新版本。安装后打开终端时，它会在打开的选项卡中通过 PowerShell 作为默认配置文件启动。[官方文档](https://docs.microsoft.com/zh-cn/windows/terminal/get-started)

## Notes
```
wsl --list --verbose
wsl --set-version <distribution name> <versionNumber>
wsl --set-default-version 2
wsl -u root

ubuntu1804.exe config --default-user root
```

## 参考资料：
* [WSL 文档](https://docs.microsoft.com/zh-cn/windows/wsl/)
* [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
* [Docker Desktop WSL 2 backend](https://docs.docker.com/docker-for-windows/wsl/)