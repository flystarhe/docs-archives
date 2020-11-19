# Tsung

## Tcp Server
```python
import socket
import datetime
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 8888))
s.listen(10)
while True:
    conn, addr = s.accept()
    data = conn.recv(1024)
    conn.close()
    print(datetime.datetime.now(), addr, type(data), data.decode("utf8"))
    if data.decode("utf8").startswith("quit"):
        s.shutdown(2)
        s.close()
        break
```

## Tcp Client
```python
import socket
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect(("127.0.0.1", 8888))
c.send("你好,abc".encode("utf8"))
#c.recv(1024).decode("utf8")
c.close()
```

## Docker
```
docker pull ubuntu:16.04 && \
docker run --name test --hostname xlab -it -p 8091:8091 ubuntu:16.04 bash

apt-get update && \
apt-get install -y vim erlang tsung python3.5

##打开新终端
docker exec -it test bash
python3.5 server.py

##配置文件范例
##local: /usr/share/doc/tsung/examples
##https: github.com/processone/tsung/tree/develop/examples
tsung -f raw.xml start

##查看记录
tail -f /root/.tsung/log/20180705-0507/tsung.log
```

## raw.xml
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tsung SYSTEM "/usr/share/tsung/tsung-1.0.dtd">
<tsung loglevel="debug" dumptraffic="true" version="1.0">

  <clients>
    <!-- host="$hostname" -->
    <client host="xlab" maxusers="2"/>
  </clients>

  <servers>
    <server host="127.0.0.1" port="8888" type="tcp"></server>
  </servers>

  <load>
    <arrivalphase phase="1" duration="1" unit="minute">
      <users maxnumber="10" arrivalrate="2" unit="second"></users>
    </arrivalphase>
  </load>

  <options>
    <option name="file_server" id="userdb" value="users.txt"/>
  </options>

 <sessions>
  <session probability="100" name="raw" type="ts_raw">
    <setdynvars sourcetype="file" fileid="userdb" delimiter=";">
      <var name="username"/>
      <var name="password"/>
    </setdynvars>

    <transaction name="open">
      <request subst="true"> <raw data="%%_username%%:%%_password%%" ack="local"></raw> </request>
    </transaction>

    <thinktime value="1"/>

  </session>
 </sessions>
</tsung>
```

## users.txt
```
何剑;none
用户名;密码
username;password
```
