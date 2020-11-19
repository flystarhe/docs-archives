# Argparse

## args
参数设置示例:
```
import argparse
def example():
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--task", required=True)
    parser.add_argument("--opt1", action="store_true")
    parser.add_argument("--opt2", action="store_false")
    parser.add_argument("--mode", default="png", type=str)
    parser.set_defaults(opt2=False)
    #如果不提供参数则从`sys.argv`读取参数
    args = parser.parse_args(["--task", "train"])
    return args
```

## toDict
`Namespace`转字典:
```
def example_to_dict():
    args = example()
    return args.__dict__

import json
import codecs
dataset = example_to_dict()
with codecs.open("dataset.txt", "w", "utf-8") as writer:
    writer.write(json.dumps(dataset, indent=2))
```

用字典初始化`Namespace`:
```
def example_from_dict():
    opts = example_to_dict()
    return argparse.Namespace(**opts)
```
