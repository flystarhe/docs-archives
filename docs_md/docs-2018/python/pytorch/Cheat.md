# Cheat
```python
torch.__version__
torch.cuda.is_available()

torch.version.cuda
torch.backends.cudnn.version()
torch.cuda.get_device_name(0)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

## 环境变量
```python
# 在命令行指定
!CUDA_VISIBLE_DEVICES=0,2,3 PYTHONPATH=`pwd`/libs:/data/libs python script.py
# 在代码中指定
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

## 释放GPU
```python
# 在代码中
torch.cuda.empty_cache()
# 在命令行
ps aux | grep python
kill -9 [pid]
# 或者重置
nvidia-smi --gpu-reset -i [gpu_id]
```

## Tensor
从只包含一个元素的张量中提取值：
```python
value = tensor.item()
```

这在训练时统计`loss`的变化过程中特别有用。否则这将累积计算图，使GPU存储占用量越来越大。

拼接张量：
```python
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)
```

注意`torch.cat`和`torch.stack`的区别在于`torch.cat`沿着给定的维度拼接，而`torch.stack`会新增一维。

## 参考资料：
- [PYTORCH CHEAT SHEET](https://pytorch.org/tutorials/beginner/ptcheat.html)
- [PyTorch Cookbook](https://zhuanlan.zhihu.com/p/59205847)