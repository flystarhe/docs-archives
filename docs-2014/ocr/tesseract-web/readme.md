title: Tess4j and tomcat
date: 2016-01-20
tags: [OCR,Tesseract]
---
前面探索了2种使用Tesseract-OCR引擎的方法，命令行和java工程。本文的任务则是把后一种方法搬到Tomcat上去，也就是Web工程。本实施例实践为`on windows`，若需要部署到linux，请参考上一篇博文。

<!--more-->
## web project
打开`eclipse-jee`新建`Dynamic Web Project`，完成`Project name`和`Target runtime`项的内容，连续两次`next`进入`Web Module`界面，勾选`Generate web.xml deployment descriptor`项，点击`Finish`完成。

在工程中找到`WebContent`，右键`new > JSP file`创建`index.jsp`，修改文件中编码参数为`utf-8`。

在`<body></body>`标签内随意键入字符，然后`Run As > Run on Server`对话框，设置服务器相关项，点击`Finish`运行。

## upload jsp
这里采用apache的common-fileupload文件上传组件。这个组件的jar包可以在[struts的lib](http://www.us.apache.org/dist/struts/2.3.24/struts-2.3.24-lib.zip)中找到。common-fileupload依赖于common-io，所以还需要下载这个包。然后新建jsp页面如下：

    #upload.jsp#
    <%@ page language="java" contentType="text/html; charset=utf-8" pageEncoding="utf-8"%>
    <!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
    <html>
    <head>
    <title>上传文件</title>
    </head>
    <body>
        提交需要 tesseract-ocr 识别的图片。<br/><br/>
        <form action="${pageContext.request.contextPath}/servlet/UploadHandleServlet" enctype="multipart/form-data" method="post">
            上传文件：<input type="file" name="file1"><br/><br/>
            <input type="submit" value="提交">
        </form>
    </body>
    </html>
    #upload-message.jsp#
    <%@ page language="java" contentType="text/html; charset=utf-8" pageEncoding="utf-8"%>
    <!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
    <html>
    <head>
    <title>消息提示</title>
    </head>
    <body>
        tesseract-ocr output:<br/><br/>
        ${message}
        <br/><br/>
        ${ipath}
        <br/><br/>
        <a href="${pageContext.request.contextPath}/upload.jsp">返回：上传文件</a>
    </body>
    </html>

## servlet
在工程`Java Resources > src`右键`new > servlet`，`Java package`项输入`com.flystarhe`，`Class name`项输入`UploadHandleServlet`，单击`next`，编辑最下方的`URL mappings`为`/servlet/UploadHandleServlet`，单击`next`，勾选doGet和doPost方法，然后点击`Finish`。代码如下：

    package com.flystarhe;
    import java.io.File;
    import java.io.FileOutputStream;
    import java.io.IOException;
    import java.io.InputStream;
    import java.text.SimpleDateFormat;
    import java.util.Date;
    import java.util.List;
    import javax.servlet.ServletException;
    import javax.servlet.annotation.WebServlet;
    import javax.servlet.http.HttpServlet;
    import javax.servlet.http.HttpServletRequest;
    import javax.servlet.http.HttpServletResponse;
    import org.apache.commons.fileupload.FileItem;
    import org.apache.commons.fileupload.disk.DiskFileItemFactory;
    import org.apache.commons.fileupload.servlet.ServletFileUpload;
    import net.sourceforge.tess4j.*;
    /**
     * Servlet implementation class UploadHandleServlet
     */
    @WebServlet(description = "servlet", urlPatterns = { "/servlet/UploadHandleServlet" })
    public class UploadHandleServlet extends HttpServlet {
        private static final long serialVersionUID = 1L;
        /**
         * @see HttpServlet#HttpServlet()
         */
        public UploadHandleServlet() {
            super();
            // TODO Auto-generated constructor stub
        }
        /**
         * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
         */
        protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
            // TODO Auto-generated method stub
            String iPath = this.getServletContext().getRealPath("/WEB-INF/upload");
            File iFile = new File(iPath);
            if(!iFile.exists() && !iFile.isDirectory()) {
                iFile.mkdir();
            }
            String iMessage = "";
            String iName = "";
            try {
                DiskFileItemFactory iFactory = new DiskFileItemFactory();
                ServletFileUpload iUpload = new ServletFileUpload(iFactory);
                iUpload.setHeaderEncoding("UTF-8");
                List<FileItem> iList = iUpload.parseRequest(request);
                for(FileItem item : iList) {
                    iName = item.getName(); // upload file
                    if(item.isFormField() || iName==null || iName.trim().equals("")) {
                        continue;
                    }
                    SimpleDateFormat dateFormater = new SimpleDateFormat("mmdd-HHmm-ss");
                    iName = iPath + "\\" + dateFormater.format(new Date()) + iName.substring(iName.lastIndexOf("."));
                    InputStream iStream = item.getInputStream();
                    FileOutputStream oStream = new FileOutputStream(iName);
                    byte buffer[] = new byte[1024];
                    int len = 0;
                    while((len=iStream.read(buffer))>0) {
                        oStream.write(buffer, 0, len);
                    }
                    iStream.close();
                    oStream.close();
                    item.delete();
                    File imageFile = new File(iName); // ocr working
                    ITesseract instance = new Tesseract();
                    instance.setDatapath(this.getServletContext().getRealPath("/WEB-INF/tessdata"));
                    instance.setLanguage("eng+chi_sim");
                    iMessage = instance.doOCR(imageFile);
                }
            } catch(Exception e) {
                iMessage = "文件上传失败！";
                e.printStackTrace();
            }
            request.setAttribute("message", iMessage);
            request.setAttribute("ipath", iName);
            request.getRequestDispatcher("/upload-message.jsp").forward(request, response);
        }
        /**
         * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
         */
        protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
            // TODO Auto-generated method stub
            doGet(request, response);
        }
    }

注：保证servlet可以工作，须拷贝[tess4j](http://sourceforge.net/projects/tess4j/files/tess4j/)依赖的jars包到`WEB-INF/lib`，然后在`WEB-INF`目录下建tessdata文件夹存放语言包[chi_sim.traineddata+eng.traineddata](https://github.com/tesseract-ocr/tessdata)。

## run on server
右键`upload.jsp`执行`Run on Server`上传图片测试OCR识别。测试工作正常，然后打成WAR包发布到Tomcat服务器的webapps目录。在项目上右键`Properties`顺序执行`Export > Export... > Web > WAR file`，单击`next`，选择输出路径并检查相关设置，点击`Finish`完成。(注：如果Tomcat报错，请将依赖的jar包拷贝一份到Tomcat的lib目录)

## 参考资料：
- [JavaWeb文件上传和下载](http://www.cnblogs.com/xdp-gacl/p/4200090.html)