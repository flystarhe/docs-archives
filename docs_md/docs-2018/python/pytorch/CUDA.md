# CUDA
`torch.cuda`实现与CPU张量相同的功能,但它们利用GPU进行计算.它被懒惰地初始化,因此您可以随时导入它,并用`torch.cuda.is_available()`确定您的系统是否支持CUDA.

## 使用设备
`torch.device`表示`torch.Tensor`将要分配的设备的对象.`torch.device`包含一个设备类型(`"cpu"/"cuda"`)和可选的设备的序号.如果设备序号不存在,则表示设备类型的当前设备.例如,`"cuda"`等价于`"cuda:X"`,`X`是结果`torch.cuda.current_device()`.`torch.Tensor`的设备可以通过访问`Tensor.device`属性.
```python
torch.device('cpu')
# device(type='cpu')
torch.device('cuda:0')
# device(type='cuda', index=0)
torch.device('cuda', 0)
# device(type='cuda', index=0)
torch.randn((1,3), dtype=torch.float32, device='cuda:0').device
# device(type='cuda', index=0)
torch.Tensor([[1, 2, 3]]).to(torch.device('cuda:0'), torch.int8)
# tensor([[ 1,  2,  3]], dtype=torch.int8, device='cuda:0')
```

以下是等效的:
```python
torch.randn((2,3), device=torch.device('cuda:1'))
torch.randn((2,3), device='cuda:1')
torch.randn((2,3), device=1)  # legacy
```

## 基本操作
返回可用的GPU数:
```python
torch.cuda.device_count()
# 1
```

返回当前所选设备的索引:
```python
torch.cuda.current_device()
# 0
```

设置当前设备,不鼓励使用此功能.最好使用`CUDA_VISIBLE_DEVICES`环境变量:
```python
torch.cuda.set_device(0)
#CUDA_VISIBLE_DEVICES=1 python script.py
#CUDA_VISIBLE_DEVICES=1           Only device 1 will be seen
#CUDA_VISIBLE_DEVICES=0,1         Devices 0 and 1 will be visible
#CUDA_VISIBLE_DEVICES="0,1"       Same as above, quotation marks are optional
#CUDA_VISIBLE_DEVICES=0,2,3       Devices 0, 2, 3 will be visible; 1 is masked
#CUDA_VISIBLE_DEVICES=""          No GPU will be visible
```

获取设备的名称:
```python
n = torch.cuda.device_count()
[torch.cuda.get_device_name(i) for i in range(n)]
# ['GeForce GTX 1080 Ti']
```

## 随机数生成器
设置种子以生成GPU的随机数:
```
torch.cuda.manual_seed(1234)     #当前GPU
torch.cuda.manual_seed_all(1234) #所有GPU
```

返回当前GPU的当前随机种子:
```
torch.cuda.initial_seed()
# 1234
```
