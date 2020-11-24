# Pixel Convolution

## pixel shuffle upscale
重新排列张量形状`(*, C \times r^2, H, W)`到形状`(*, C, H \times r,W \times r)`。这对于实现有效的子像素卷积非常有用`1/r`。

PyTorch已经为我们实现`torch.nn.PixelShuffle(upscale_factor)`，参数`upscale_factor`增加空间分辨率的因子，上文中的`r`。用法：
```python
pixel_shuffle = nn.PixelShuffle(3)
input = torch.randn(1, 9, 4, 4)
output = pixel_shuffle(input)
print(output.size())
## torch.Size([1, 1, 12, 12])
```

自己实现：
```python
def pixel_shuffle_upscale(input, scale_factor):
    b = input.size(0)
    c = input.size(1)
    h = input.size(2)
    w = input.size(3)
    scale_factor_squared = scale_factor * scale_factor

    oc = c // scale_factor_squared
    oh = h * scale_factor
    ow = w * scale_factor

    input_reshaped = input.reshape(b, oc, scale_factor, scale_factor, h, w)
    return input_reshaped.permute(0, 1, 4, 2, 5, 3).reshape(b, oc, oh, ow)
```

## pixel shuffle downscale
重新排列张量形状`(*, C, H \times r,W \times r)`到形状`(*, C \times r^2, H, W)`。这是上节的逆操作，在我的MRI去鬼影工作中非常有用。

PyTorch中没有提供现成的函数，自己实现：
```python
def pixel_shuffle_downscale(input, scale_factor):
    b = input.size(0)
    c = input.size(1)
    h = input.size(2)
    w = input.size(3)
    scale_factor_squared = scale_factor * scale_factor

    oc = c * scale_factor_squared
    oh = h // scale_factor
    ow = w // scale_factor

    input_reshaped = input.reshape(b, c, oh, scale_factor, ow, scale_factor)
    return input_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)
```

## test
```python
import torch
import torch.nn as nn

scale_factor = 2

input = torch.randint(0, 9, (1, 4, 2, 2))

pixel_shuffle = nn.PixelShuffle(scale_factor)
a = pixel_shuffle(input)

b = pixel_shuffle_upscale(input, scale_factor)
print((a - b).abs().sum())

c = pixel_shuffle_downscale(b, scale_factor)
print((c - input).abs().sum())
```
