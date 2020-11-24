title: Tesseract-ocr and tess4j
date: 2016-01-19
tags: [OCR,Tesseract]
---
Tesseract是一个OCR引擎，在1985年到1995年由HP实验室开发，现在在Google。从3.0开始支持中文，这标志着现在有自由的中文OCR软件了。本文先尝试在CentOS下安装tesseract，然后基于tess4j构建java工程。

<!--more-->
## CentOS: 安装
检查旧版本，如果存在先卸载：

    $ rpm -qa | grep -i tesseract
    $ rpm -e --nodeps tesseract
    $ rpm -qa | grep -i leptonica
    $ rpm -e --nodeps leptonica

添加epel源：

    $ rpm -Uvh http://mirrors.ustc.edu.cn/epel/epel-release-latest-7.noarch.rpm
    $ rpm -Uvh http://mirrors.ustc.edu.cn/epel/epel-release-latest-6.noarch.rpm

安装tesseract-ocr及语言包：

    $ yum -y install gcc gcc-c++ make automake autoconf libtool libpng libjpeg libtiff openjpeg zlib
    $ yum search tesseract
    $ yum -y install tesseract
    $ yum -y install tesseract-langpack-chi_sim tesseract-langpack-chi_tra
    $ rpm -qi tesseract
    $ rpm -ql tesseract

验证tesseract-ocr，保证`1.jpg`测试文件已就绪：

    $ tesseract 1.jpg 1.1 -l eng+chi_sim
    $ cat 1.1.txt

## ubuntu: 安装
开启ssh远程连接：

    $ sudo apt-get install openssh-server
    $ /etc/init.d/ssh start

安装相关依赖：

    $ sudo apt-get install autoconf automake libtool
    $ sudo apt-get install libpng12-dev
    $ sudo apt-get install libjpeg62-dev
    $ sudo apt-get install libtiff4-dev
    $ sudo apt-get install zlib1g-dev
    $ sudo apt-get install libicu-dev
    $ sudo apt-get install libpango1.0-dev
    $ sudo apt-get install libcairo2-dev

安装tesseract-ocr及语言包：

    $ sudo apt-cache search tesseract
    $ sudo apt-get install tesseract-ocr
    $ sudo apt-get install tesseract-ocr-chi-sim

验证tesseract-ocr，保证`1.jpg`测试文件已就绪：

    $ tesseract 1.jpg 1.1 -l eng+chi_sim
    $ cat 1.1.txt

## for java on linux
启动idea新建sbt项目，scala代码如下：

    package cn.com.cetc.lab.ocr
    import java.io.{FileReader, BufferedReader}
    import java.util
    /**
      * Created by jian on 2016/2/19.
      */
    class Tesseract {
      val name = "Tesseract"
    }
    object Tesseract {
      def doOCR(ifile: String, iconf: String = "eng+chi_sim", itag: String = "linux"): String = {
        return itag match {
          case "linux" => doOCR4lin(ifile, iconf)
          case "windows" => doOCR4win(ifile, iconf)
          case _ => "parameter error!"
        }
      }
      def doOCR4win(ifile: String, iconf: String = "eng+chi_sim"): String = {
        var res = ""
        val xcmd = new util.ArrayList[String]()
        xcmd.add("cmd")
        xcmd.add("/c")
        xcmd.add("tesseract")
        xcmd.add(ifile)
        xcmd.add("tmp")
        xcmd.add("-l")
        xcmd.add(iconf)
        val xpb = new ProcessBuilder()
        xpb.command(xcmd)
        xpb.redirectErrorStream(true)
        val xpc = xpb.start()
        val tim = xpc.waitFor()
        if(tim == 0) {
          val xin = new BufferedReader(new FileReader("tmp.txt"))
          var line = xin.readLine()
          while(line != null) {
            res += line + "\r\n"
            line = xin.readLine()
          }
        } else {
          res += "sorry, error code is: " + tim
        }
        return res
      }
      def doOCR4lin(ifile: String, iconf: String = "eng+chi_sim"): String = {
        var res = ""
        val xcmd = new util.ArrayList[String]()
        xcmd.add("tesseract")
        xcmd.add(ifile)
        xcmd.add("tmp")
        xcmd.add("-l")
        xcmd.add(iconf)
        val xpb = new ProcessBuilder()
        xpb.command(xcmd)
        xpb.redirectErrorStream(true)
        val xpc = xpb.start()
        val tim = xpc.waitFor()
        if(tim == 0) {
          val xin = new BufferedReader(new FileReader("tmp.txt"))
          var line = xin.readLine()
          while(line != null) {
            res += line + "\n"
            line = xin.readLine()
          }
        } else {
          res += "sorry, error code is " + tim
        }
        return res
      }
    }

## for java on windows
启动Eclipse新建java项目，并将[Tess4J-3.0-src.zip](http://sourceforge.net/projects/tess4j/files/tess4j/)项目lib中所有文件和dist中`tess4j-3.0.jar`拷贝到项目中，然后`Build Path > Configure Build Path > Add External JARs`添加`JAR`文件。新建类`TesseractExample`测试，请确认项目根目录存在测试图片`1.jpg`。

    package test4ocr;
    import java.io.File;
    import net.sourceforge.tess4j.*;
    public class TesseractExample {
        public static void main(String[] args) {
            try {
                File imageFile = new File("1.jpg");
                ITesseract instance = new Tesseract();
                instance.setDatapath("E:/jars-ocr/tessdata");
                instance.setLanguage("eng+chi_sim");
                String res = instance.doOCR(imageFile);
                System.out.println("ocr_txt:\n" + res);
            } catch(Exception e) {
                System.err.println(e.getMessage());
            }
        }
    }

注：目录`E:/jars-ocr/tessdata`存放有相关语言包，测试来看图片识别效果非常不错。

## 参考资料：
- [tesseract github](https://github.com/tesseract-ocr/tesseract)
- [tesseract wiki](https://github.com/tesseract-ocr/tesseract/wiki/Compiling)
- [leptonica home](http://www.leptonica.com/)
- [tess4j home](http://sourceforge.net/projects/tess4j)
- [tess4j github](https://github.com/nguyenq/tess4j)
- [tess4j readme](https://github.com/nguyenq/tess4j/blob/master/src/main/resources/readme.html)
- [Java OCR tesseract Java代码实现](http://blog.csdn.net/lmj623565791/article/details/23960391)