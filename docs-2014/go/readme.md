title: Go | Quick Start
date: 2017-10-20
tags: [Go]
---
Go是一种并发的,带垃圾回收的,快速编译的,支持网络与多核计算的语言.它结合了解释型语言的游刃有余,动态类型语言的开发效率,以及静态类型的安全性.

<!--more-->
有一个你需要记在心里的事情是,Go语言是为大型软件设计的.我们都喜欢程序简洁清晰,但对于一个由很多程序员一起开发的大型软件,维护成本的增加很难让程序简洁.异常捕捉模式的错误处理方式的一个很有吸引力的特点是,它非常适合小程序.

Go语言的返回错误方式,不可否认,对于调用者不是很方便,但这样做会让程序中可能会出错的地方显的很明显,对于小程序来说,你可能只想打印出错误,退出了程序.对于一些很精密的程序,根据异常不同,来源不同,程序会做出不同的反应,这很常见,这种情况中`try + catch`的方式相对于错误返回模式显得冗长.

在我们的Python里面一个10行的代码放到Go语言里很可能会更冗长,Go语言主要不是针对10行规模的程序.

## golang
安装Go,可以从[官方网站](https://golang.org/doc/install)为您的操作系统获取二进制发行版,比如[go1.9.1.linux-amd64.tar.gz](#).有必要说明的是,Go官网是需要翻墙的,我用的是[XX-net/XX-Net](https://github.com/XX-net/XX-Net).解压到`/usr/local`,创建一个Go树`/usr/local/go`,例如:

    sudo tar -C /usr/local -xzf go1.9.1.linux-amd64.tar.gz

添加`/usr/local/go/bin`到`PATH`环境变量.将此行添加到`/etc/profile`:

    export GOROOT=/usr/local/go
    export PATH=$GOROOT/bin:$PATH

记得执行`source /etc/profile`.如果您在Mac上使用Homebrew,`brew install go`效果很好.如果您是Ubuntu,`sudo apt install golang-go`也很方便.检查:

    $ go version
    go version go1.9.1 linux/amd64

### test
在工作目录构建简单的程序来检查Go是否正确安装,如`$HOME/go/hello.go`:
```go
package main

import "fmt"

func main() {
    fmt.Printf("hello, world\n")
}
```

使用Go工具构建它:

    $ cd $HOME/go/
    $ go build hello.go

上面的命令将构建一个可执行文件,执行它来查看问候语:

    $ ./hello
    hello, world

如果单纯想测试一下,可以使用`go run`(不构建出可执行文件):

    $ cd $HOME/go/
    $ go run hello.go
    hello, world

### uninstall
如果您从旧版本的Go升级,您必须先删除现有版本.要从系统中删除现有的Go安装,请删除该Go目录,通常是`/usr/local/go`.如果通过`apt`安装,则执行`sudo apt purge golang-go`.

## web server
```go
// filename: server.go
package main

import "net/http"

func main() {
    http.HandleFunc("/", hello)
    http.ListenAndServe(":8080", nil)
}

func hello(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("你好!"))
}
```

首先,我们需要从标准库导入`net/http`包.然后,在`main`函数中为web服务器的根路径安装一个`handler`函数:
```go
http.HandleFunc("/", hello)
```

[http.HandleFunc](http://golang.org/pkg/net/http/#HandlerFunc)在默认的HTTP路由器上运行,官方称之为[ServeMux](http://golang.org/pkg/net/http/#ServeMux).`hello`是一个[http.HandleFunc](http://golang.org/pkg/net/http/#HandlerFunc),意味着它有一个特定的类型签名.每当新的请求进入,服务器将产生一个执行`hello`函数的新`goroutine`,而`hello`函数只需使用[http.ResponseWriter](http://golang.org/pkg/net/http/#ResponseWriter)将响应写入客户端.由于`http.ResponseWriter.Write`采用更通用`[]byte`作为参数,因此我们对字符串进行简单的类型转换:
```go
func hello(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("你好!"))
}
```

最后,我们在端口8080上启动HTTP服务器,并通过[http.ListenAndServe](http://golang.org/pkg/net/http/#ListenAndServe)启动默认的`ServeMux`.像以前一样编译和运行:

    $ go build server.go
    $ ./server

在另一个终端或浏览器中,发出HTTP请求:

    $ curl http://localhost:8080
    你好!

## more routes
我们可以做一些比只是打招呼更有趣的事情.我们把一个城市作为输入,访问一个天气API,并转发温度的响应.该[OpenWeatherMap](http://openweathermap.org/)提供了一个[简单且免费的API](http://openweathermap.org/api)用于目前的预测信息。[注册](https://home.openweathermap.org/users/sign_up)一个免费帐户以获取API密钥.`OpenWeatherMap`的API可以按城市查询.它返回这样的回应:
```
{
    "name": "Tokyo",
    "coord": {
        "lon": 139.69,
        "lat": 35.69
    },
    "weather": [
        {
            "id": 803,
            "main": "Clouds",
            "description": "broken clouds",
            "icon": "04n"
        }
    ],
    "main": {
        "temp": 296.69,
        "pressure": 1014,
        "humidity": 83,
        "temp_min": 295.37,
        "temp_max": 298.15
    }
}
```

Go是静态类型的语言,所以我们应该创建一个响应格式的结构.我们不需要捕获每一个信息,只需要我们关心的东西.我们定义一个结构来表示天气API返回的数据:
```go
type weatherData struct {
    Name string `json:"name"`
    Main struct {
        Kelvin float64 `json:"temp"`
    } `json:"main"`
}
```

`type`关键字定义了一种新的类型,我们称之为`weatherData`,并声明为一个结构体.结构中的每个字段都有一个名称`Name,Main`,一个类型`string`另一个匿名`struct`,以及所谓的标签.标签就像元数据,并允许我们使用[encoding/json](http://golang.org/pkg/encoding/json)包直接将API的响应解组成我们的结构体.与Python或Ruby等动态语言相比,它的打字比较多,但它使我们成为安全类型的非常理想的属性.我们已经定义了结构,现在我们需要定义一个填充它的方法:
```go
func query(city string) (weatherData, error) {
    resp, err := http.Get("http://samples.openweathermap.org/data/2.5/weather?q=London,uk&appid=b1b15e88fa797225412429c1c50c122a1")
    if err != nil {
        return weatherData{}, err
    }

    defer resp.Body.Close()

    var d weatherData

    if err := json.NewDecoder(resp.Body).Decode(&d); err != nil {
        return weatherData{}, err
    }

    return d, nil
}
```

该函数使用一个表示城市的字符串,并返回一个`weatherData`结构和一个错误.这是Go中的基本错误处理习语.函数编码行为,行为通常可能会失败.对于我们来说,针对`OpenWeatherMap`的GET请求可能由于任何原因而失败,并且返回的数据可能不是我们期望的.在任一种情况下,我们向客户端返回一个非零错误,将以一种在调用上下文中有意义的方式来处理它.

如果`http.Get`成功,我们推迟一个关闭响应体的调用,当我们离开函数范围(从查询函数返回时)将执行,并且是一种优雅的资源管理形式.同时,我们分配一个`weatherData`结构,并使用一个`json.Decoder`将响应体直接从我们的结构体中解组.

除此之外,`json.NewDecoder`利用了Go的优雅功能,它们是接口.解码器没有具体的HTTP响应体,相反,它需要一个`io.Reader`接口,`http.Response.Body`恰好满足.解码器提供一个行为`解码`,它只通过调用满足其他行为的类型的方法`Read`.

最后,如果解码成功,我们将`weatherData`返回给调用者,其中一个零错误表示成功.现在让我们把这个函数连接到请求处理程序:
```go
http.HandleFunc("/weather/", func(w http.ResponseWriter, r *http.Request) {
    city := strings.SplitN(r.URL.Path, "/", 3)[2]

    data, err := query(city)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json; charset=utf-8")
    json.NewEncoder(w).Encode(data)
})
```

在这里,我们定义了内联处理程序,而不是单独的函数.我们使用`strings.SplitN`把所有放在路径`/weather/`之后的东西都视为城市.我们进行查询,如果有错误,我们将使用`http.Error`帮助函数将其报告给客户端.我们需要返回,完成HTTP请求.否则,我们告诉客户我们要发送JSON数据,并使用`json.NewEncoder`直接对`weatherData`进行JSON编码.完整的程序:
```go
// filename: server.go
package main

import (
    "encoding/json"
    "net/http"
    "strings"
)

func main() {
    http.HandleFunc("/hello", hello)

    http.HandleFunc("/weather/", func(w http.ResponseWriter, r *http.Request) {
        city := strings.SplitN(r.URL.Path, "/", 3)[2]

        data, err := query(city)
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }

        w.Header().Set("Content-Type", "application/json; charset=utf-8")
        json.NewEncoder(w).Encode(data)
    })

    http.ListenAndServe(":8080", nil)
}

func hello(w http.ResponseWriter, r *http.Request) {
    w.Write([]byte("你好!"))
}

func query(city string) (weatherData, error) {
    resp, err := http.Get("http://samples.openweathermap.org/data/2.5/weather?appid=b1b15e88fa797225412429c1c50c122a1&q=" + city)
    if err != nil {
        return weatherData{}, err
    }

    defer resp.Body.Close()

    var d weatherData

    if err := json.NewDecoder(resp.Body).Decode(&d); err != nil {
        return weatherData{}, err
    }

    return d, nil
}

type weatherData struct {
    Name string `json:"name"`
    Main struct {
        Kelvin float64 `json:"temp"`
    } `json:"main"`
}
```

像以前一样编译和运行:

    $ go build server.go
    $ ./server

在另一个终端或浏览器中,发出HTTP请求:

    $ curl http://localhost:8080/weather/London,uk
    {"name":"Tokyo","main":{"temp":295.9}}

## 参考资料:
- [golang-doc](https://golang.org/doc/install)
- [how-i-start-go](http://howistart.org/posts/go/1/index.html)