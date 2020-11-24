title: GPS经纬度距离计算
date: 2014-12-15
tags: [LBS,Distance,PHP]
---
最近因为工作问题需要计算距离，用户或者商户反馈回来的位置数据都是经纬度数据。所以琢磨了一下如何计算距离，虽然网上有很多函数实现，但其实都不怎么适合我的场景，因为我需要在数据库中计算距离和排序，所以我希望的距离只需要支持排序，所以并不需要真实计算出距离，但一定要快。

<!--more-->
## 精准
下面给一个PHP的高精度实现:
```php
//起始坐标($lat1,$lon1)
//目标坐标($lat2,$lon2)
$lat1 = 30;
$lon1 = 104;
$lat2 = 30.1;
$lon2 = 104.1;
function getdist($lat1,$lon1,$lat2,$lon2){
    $x = M_PI/180;
    $a = abs($lat1-$lat2)*$x;
    $b = abs($lon1-$lon2)*$x;
        return 2*asin(sqrt(pow(sin($a/2),2)+cos($lat1*$x)*cos($lat2*$x)*sin($b/2),2)))*6378.137;
}
```

## 快速
我们看重的保证排序正确和快速，下面是略微牺牲精度的实现:
```php
//在假设地球是一个圆球的条件下(半径R=6378.137)
//纬度变化1度对应的距离是固定的(R*pi/180)
//精度变化1度对应的距离与纬度有关(R*pi/180*cos(纬度))
//起始坐标($lat1,$lon1)
//目标坐标($lat2,$lon2)
$lat1 = 30;
$lon1 = 104;
$lat2 = 30.1;
$lon2 = 104.1;
$coe1 = 111.31949079327;//R*pi/180
$coe2 = 0.017453292519943;//pi/180
$dlat = $coe1;//R*pi/180
$dlon = $coe1*cos($lat1*$coe2);//R*pi/180*cos(纬度)
$distance = sqrt(pow(($lat1-$lat2)*$dlat,2)+pow(($lon1-$lon2)*$dlon,2));
//优化以上表达式
$distance = $coe1*sqrt(pow($lat1-$lat2,2)+pow(($lon1-$lon2)*cos($lat1*$coe2),2));
//其实真正影响排序的在sqrt里面的内容
$sort_num = pow($lat1-$lat2,2)+pow(($lon1-$lon2)*cos($lat1*$coe2),2);
$sort_num = pow($lat1-$lat2,2)+pow(($lon1-$lon2),2)*pow(cos($lat1*$coe2),2);
//end: 所以按照上式计算排序就可以了
//前端显示距离只需简单运算即可|移动开发真的不需要纠结设备性能
$distance = $coe1*sqrt($sort_num);
```

## 百度坐标到GPS转换
熟悉百度地图API的朋友都知道，百度提供了GPS到百度坐标的接口，但需要把百度坐标转换到GPS该怎么办呢？这个转换算法百度是不公开的，而且也没有提供接口，那就只能耍点小聪明了。(虽然精度略有损失，但绝大多数应用场景已无碍)
百度坐标和GPS坐标转换在很近的距离时偏差非常接近。
假设你有百度坐标:`x1 = 116, y1 = 39`
把这个坐标当成GPS坐标通过接口获得它的百度坐标:`x2 = 116.4, y2 = 39.1`
通过如下计算就可以得到GPS坐标:`x = 2*x1-x2, y = 2*y1-y2`。