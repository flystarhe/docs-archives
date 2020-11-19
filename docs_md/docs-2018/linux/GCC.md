# GCC

## CentOS
```
yum -y install epel-release
yum -y update
yum install -y gcc gcc-c++ glibc-static
## 开始安装7.3.0
tar -zxf gcc-7.3.0.tar.gz
cd gcc-7.3.0/
./contrib/download_prerequisites
./configure --prefix=/usr/local/gcc --enable-checking=release --enable-languages=c,c++ --disable-multilib
make -j8
make install
gcc --version
echo $'export PATH=/usr/local/gcc/bin:$PATH' >> /etc/profile
source /etc/profile
gcc --version
```

## Ubuntu
```
sudo apt update
sudo apt install build-essential
## 该命令将安装一堆新包，包括gcc，g++和make
## Ubuntu 18.04存储库中可用的默认GCC版本是7.4.0
tar -zxf gcc-7.3.0.tar.gz
cd gcc-7.3.0/
./contrib/download_prerequisites
./configure --prefix=/usr/local/gcc --enable-checking=release --enable-languages=c,c++ --disable-multilib
make -j8
make install
gcc --version
echo $'export PATH=/usr/local/gcc/bin:$PATH' >> /etc/profile
source /etc/profile
gcc --version
```
