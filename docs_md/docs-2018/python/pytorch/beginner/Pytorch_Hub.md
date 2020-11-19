# Pytorch Hub
Pytorch Hub是经过预先训练的模型资料库，旨在促进研究的可重复性。

## 发布模型
Pytorch Hub支持通过添加简单`hubconf.py`文件将预训练的模型（模型定义和预训练的权重）发布到github存储库。`hubconf.py`可以有多个`entrypoint`。每个`entrypoint`都定义为python函数（例如：您要发布的经过预先训练的模型）。
```python
def entrypoint_name(*args, **kwargs):
    # args & kwargs are optional, for models which take positional/keyword arguments.
    ...
```

## 实现入口点
如果我们在resnet18中扩展实现，下面的代码片段指定了模型的入口点`pytorch/vision/hubconf.py`。在这里，我们仅以扩展版本为例来说明其工作原理。您可以在[pytorch/vision repo](https://github.com/pytorch/vision/blob/master/hubconf.py)中看到完整的脚本。
```python
dependencies = ['torch']
from torchvision.models.resnet import resnet18 as _resnet18

# resnet18 is the name of entrypoint
def resnet18(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet18(pretrained=pretrained, **kwargs)
    return model
```

- dependencies是加载模型所需的软件包名称的列表。请注意，这可能与训练模型所需的依赖项稍有不同。
- 入口点函数可以返回模型`nn.module`，也可以使用辅助工具来使用户工作流程更流畅，例如令牌生成器。
- 带有下划线前缀的可调用项被视为辅助函数，不会在中显示`torch.hub.list()`。
- 预训练的权重既可以存储在github存储库中，也可以由加载`torch.hub.load_state_dict_from_url()`。
- 如果小于2GB，建议将其附加到[project release](https://help.github.com/en/articles/distributing-large-binaries)中，然后使用这个url。

```python
if pretrained:
    # For checkpoint saved in local github repo, e.g. <RELATIVE_PATH_TO_CHECKPOINT>=weights/save.pth
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, <RELATIVE_PATH_TO_CHECKPOINT>)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    # For checkpoint saved elsewhere
    checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))
```

## 从Hub加载模型
Pytorch Hub提供了便捷的API来探索Hub中的所有可用模型，可通过浏览`torch.hub.list()`显示`docstring`，通过`torch.hub.help()`查看示例，使用`torch.hub.load()`加载预训练的模型。
```python
# torch.hub.list(github, force_reload=False)
# github (string) : a string with format 'repo_owner/repo_name[:tag_name]'
# force_reload (bool, optional) : whether to discard the existing cache and force a fresh download
entrypoints = torch.hub.list('pytorch/vision', force_reload=True)

# torch.hub.help(github, model, force_reload=False)
# github (string) : a string with format 'repo_owner/repo_name[:tag_name]'
# model (string) : a string of entrypoint name defined in repo’s hubconf.py
torch.hub.help('pytorch/vision', 'resnet18', force_reload=True)

# torch.hub.load(github, model, *args, **kwargs)
# github (string) : a string with format 'repo_owner/repo_name[:tag_name]'
# model (string) : a string of entrypoint name defined in repo’s hubconf.py

# torch.hub.download_url_to_file(url, dst, hash_prefix=None, progress=True)
# url (string) : URL of the object to download
# dst (string) : Full path where object will be saved, e.g. /tmp/temporary_file
torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')

# torch.hub.load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False)
# url (string) : URL of the object to download
# model_dir (string, optional) : directory in which to save the object
# map_location (optional) : a function or a dict specifying how to remap storage locations (see torch.load)
state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
```

## 参考资料：
- [Pytorch Hub](https://pytorch.org/docs/stable/hub.html)