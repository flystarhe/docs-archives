# COCO
COCO有五种注释类型：用于对象检测，关键点检测，填充分割，全景分割和图像字幕。注释使用JSON存储。[COCO API](https://github.com/cocodataset/cocoapi)可用于访问和操作所有anotations。

## 数据格式
所有COCO注释共享下面相同的基本数据结构：
```
{
    "info"             : info,
    "images"           : [image],
    "annotations"      : [annotation],
    "licenses"         : [license],
}

info{
    "year"             : int,
    "version"          : str,
    "description"      : str,
    "contributor"      : str,
    "url"              : str,
    "date_created"     : datetime,
}

image{
    "id"               : int,
    "width"            : int,
    "height"           : int,
    "file_name"        : str,
    "license"          : int,
    "flickr_url"       : str,
    "coco_url"         : str,
    "date_captured"    : datetime,
}

license{
    "id"               : int,
    "name"             : str,
    "url"              : str,
}
```

## 物体检测
每个对象实例`annotation`都包含一系列字段，包括对象的类别ID和分割掩码。分割格式取决于实例是表示单个对象（iscrowd=0，在这种情况下使用多边形）还是对象集合（iscrowd=1，在这种情况下使用RLE）。请注意，单个对象（iscrowd=0）可能需要多个多边形，例如，如果被遮挡。人群注释（iscrowd=1）用于标记大组对象（例如一群人）。另外，为每个对象提供封闭边界框（框坐标从左上角图像角测量并且是0索引的）。另请参阅[检测任务](http://cocodataset.org/#detection-2018)。
```
annotation{
    "id"               : int,
    "image_id"         : int,
    "category_id"      : int,
    "segmentation"     : RLE or [polygon],
    "area"             : float,
    "bbox"             : [x,y,width,height],
    "iscrowd"          : 0 or 1,
}

categories[{
    "id"               : int,
    "name"             : str,
    "supercategory"    : str,
}]
```

## segmentation & mask
```
import numpy as np
from pycocotools import mask as maskUtils
```

多边形转RLE：
```
polygon = [1, 1, 3, 1, 3, 3, 5, 3, 5, 5, 1, 5]
ann = dict(segmentation=[polygon])

rles = maskUtils.frPyObjects(ann["segmentation"], 10, 10)
rle = maskUtils.merge(rles)
{'size': [10, 10], 'counts': b';4602N00_1'}
```

RLE转MASK：
```
mask = maskUtils.decode(rle)
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
       [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
```

MASK转RLE：
```
maskUtils.encode(mask)
{'size': [10, 10], 'counts': b';4602N00_1'}
```

ndarray转RLE：
```
imask = np.zeros((10, 10), dtype=np.uint8)
imask[1:3, 1:3] = 1
imask[3:5, 1:5] = 1
maskUtils.encode(np.asfortranarray(imask))
{'size': [10, 10], 'counts': b';4602N00_1'}
```

## 参考资料：
- [Data format](http://cocodataset.org/#format-data)