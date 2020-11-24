# Image augmentation

## Installation

### imgaug
```bash
conda config --add channels conda-forge
conda install imgaug

pip install imgaug
```

### albumentations
```bash
conda install -c conda-forge imgaug
conda install albumentations -c albumentations

pip install albumentations
```

## albu
```python
import os
import random
import cv2 as cv
import numpy as np
import albumentations as A
from skimage.color import label2rgb
from matplotlib import pyplot as plt


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2, **kwargs):
    #height, width = img.shape[:2]
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return img


def visualize_titles(img, bbox, title, color=BOX_COLOR, thickness=2, font_thickness=2, font_scale=0.35, **kwargs):
    #height, width = img.shape[:2]
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    ((text_width, text_height), _) = cv.getTextSize(title, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv.putText(img, title, (x_min, y_min - int(0.3 * text_height)), cv.FONT_HERSHEY_SIMPLEX,
               font_scale, TEXT_COLOR, font_thickness, lineType=cv.LINE_AA)
    return img


def augment_and_show(aug, image, mask=None, bboxes=[], categories=[], category_id_to_name=[], filename=None,
                     font_scale_ori=0.35, font_scale_aug=0.35, show_title=True, **kwargs):
    augmented = aug(image=image, mask=mask, bboxes=bboxes, category_id=categories)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_aug = cv.cvtColor(augmented['image'], cv.COLOR_BGR2RGB)

    for bbox in bboxes:
        visualize_bbox(image, bbox, **kwargs)

    for bbox in augmented['bboxes']:
        visualize_bbox(image_aug, bbox, **kwargs)

    if show_title:
        for bbox, cat_id in zip(bboxes, categories):
            visualize_titles(image, bbox, category_id_to_name[cat_id], font_scale=font_scale_ori, **kwargs)
        for bbox, cat_id in zip(augmented['bboxes'], augmented['category_id']):
            visualize_titles(image_aug, bbox, category_id_to_name[cat_id], font_scale=font_scale_aug, **kwargs)

    if mask is None:
        f, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(image)
        ax[0].set_title('Original image')

        ax[1].imshow(image_aug)
        ax[1].set_title('Augmented image')
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 16))

        if len(mask.shape) != 3:
            mask = label2rgb(mask, bg_label=0)
            mask_aug = label2rgb(augmented['mask'], bg_label=0)
        else:
            mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
            mask_aug = cv.cvtColor(augmented['mask'], cv.COLOR_BGR2RGB)

        ax[0, 0].imshow(image)
        ax[0, 0].set_title('Original image')

        ax[0, 1].imshow(image_aug)
        ax[0, 1].set_title('Augmented image')

        ax[1, 0].imshow(mask, interpolation='nearest')
        ax[1, 0].set_title('Original mask')

        ax[1, 1].imshow(mask_aug, interpolation='nearest')
        ax[1, 1].set_title('Augmented mask')

    f.tight_layout()
    if filename is not None:
        f.savefig(filename)

    return augmented['image'], augmented['mask'], augmented['bboxes']
```

### Image
```python
def test():
    random.seed(1234)
    image = cv.imread('albumentations-master/notebooks/images/parrot.jpg')
    aug = A.RandomBrightnessContrast(brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1.0)
    r = augment_and_show(aug, image)
    return r
```

## 参考资料：
- [imgaug](https://github.com/aleju/imgaug)
- [Albumentations](https://github.com/albumentations-team/albumentations)