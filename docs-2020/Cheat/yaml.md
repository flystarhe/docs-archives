# YAML
YAML是一种数据序列化格式，旨在提高人类可读性并与脚本语言进行交互。PyYAML是Python的YAML。[YACS](https://github.com/rbgirshick/yacs)是一个轻量级的库，用于定义和管理系统配置，例如为科学实验而设计的软件中常见的配置。这些“配置”通常涵盖概念，例如用于训练机器学习模型的超参数或可配置的模型超参数（例如卷积神经网络的深度）。由于从事科学工作，因此可重复性至关重要，因此需要一种可靠的方法来序列化实验配置。YACS使用YAML作为一种简单的，人类可读的序列化格式。

* [yaml.org](https://yaml.org/)
* [pyyaml.org](https://pyyaml.org/)
* [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)
* [github.com/rbgirshick/yacs](https://github.com/rbgirshick/yacs)

```
pip install pyyaml
pip install yacs
```

## PyYAML
`yaml.load`接受字节字符串，Unicode字符串，打开的二进制文件对象或打开的文本文件对象。
```python
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

data = yaml.load('''
a: Some Words
b: Label
c:
  - 3
  - 4
  - 5
d:
  e: 6
  f: 7
''', Loader=Loader)
text = yaml.dump(data, Dumper=Dumper, encoding='utf-8', default_flow_style=False)
print(data, '#' * 32, text.decode(encoding='utf-8'), sep='\n')
```

如果字符串或文件包含多个文档，则可以使用`yaml.load_all`函数将它们全部加载。如果需要将多个YAML文档转储到单个流中，请使用函数`yaml.dump_all`。
```python
documents = '''
---
name: The Set of Gauntlets 'Pauraegen'
description: >
    A set of handgear with sparks that crackle
    across its knuckleguards.
---
name: The Set of Gauntlets 'Paurnen'
description: >
  A set of gauntlets that gives off a foul,
  acrid odour yet remains untarnished.
---
name: The Set of Gauntlets 'Paurnimmen'
description: >
  A set of handgear, freezing with unnatural cold.
'''
for data in yaml.load_all(documents):
    print(data)
```

PyYAML允许构造任何类型的Python对象。
```python
yaml.load("""
none: [~, null]
bool: [true, false, on, off]
int: 42
float: 3.14159
list: [LITE, RES_ACID, SUS_DEXT]
dict: {hp: 13, sp: 5}
""")

{'none': [None, None], 'int': 42, 'float': 3.14159,
'list': ['LITE', 'RES_ACID', 'SUS_DEXT'], 'dict': {'hp': 13, 'sp': 5},
'bool': [True, False, True, False]}
```

## yacs
要在项目中使用YACS，首先要创建一个项目配置文件，通常称为`config.py`或`defaults.py`。该文件是所有可配置选项的一站式参考点。它应该有很好的文档记录，并为所有选项提供合理的默认值。
```python
# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.HYPERPARAMETER_1 = 0.1
# The all important scales for the stuff
_C.TRAIN.SCALES = (2, 4, 8, 16)


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as a global singleton:
# cfg = _C  # users can `from config import cfg`
```

接下来，您将创建YAML配置文件。通常您会为每个实验做一个。每个配置文件仅覆盖该实验中正在更改的选项。
```
# my_project/experiment.yaml
SYSTEM:
  NUM_GPUS: 2
TRAIN:
  SCALES: (1, 2)
```

最后，您将拥有使用config系统的实际项目代码。进行任何初始设置后，最好通过调用`freeze()`方法冻结它，以防止进行进一步修改。如下所示，通过`cfg`直接导入和访问config选项，可以将其用作全局选项集，也可以将config选项`cfg`复制并作为参数传递。
```python
# my_project/main.py
import my_project
from config import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
  cfg = get_cfg_defaults()
  cfg.merge_from_file("experiment.yaml")
  cfg.freeze()
  print(cfg)

  # Example of using the cfg as global access to options
  if cfg.SYSTEM.NUM_GPUS > 0:
    my_project.setup_multi_gpu_support()

  model = my_project.create_model(cfg)
```

您可以使用完全限定的键值对列表来更新`CfgNode`。这样可以很容易地从命令行使用替代选项。例如：
```python
cfg.merge_from_file("experiment.yaml")
# Now override from a list (opts could come from the command line)
opts = ["SYSTEM.NUM_GPUS", 8, "TRAIN.SCALES", "(1, 2, 3, 4)"]
cfg.merge_from_list(opts)
```

## syntax
YAML使用`---`分隔流内的文档。`...`表明文档的末尾而没有开始新的文档，供在通信通道中使用。`#`表示注释。YAML中有5种标量样式：普通，单引号，双引号，文字式和折叠式。标量内容可以以块形式编写，使用文字式`|`，其中每个换行符都将保留。也可以使用折叠式`>`，其中每个换行符折叠为一个空格，除非空行或更多缩进。[Chapter 2 of the YAML specification](https://yaml.org/spec/1.1/#id857168)

Sequence of Scalars
```
- Mark McGwire
- Sammy Sosa
- Ken Griffey
```

Mapping Scalars to Scalars
```
hr:  65    # Home runs
avg: 0.278 # Batting average
rbi: 147   # Runs Batted In
```

Mapping Scalars to Sequences
```
american:
  - Boston Red Sox
  - Detroit Tigers
  - New York Yankees
national:
  - New York Mets
  - Chicago Cubs
  - Atlanta Braves
```

Sequence of Mappings
```
-
  name: Mark McGwire
  hr:   65
  avg:  0.278
-
  name: Sammy Sosa
  hr:   63
  avg:  0.288
```

In-Line Nested Mapping
```
- name: Mark McGwire
  hr:   65
  avg:  0.278
- name: Sammy Sosa
  hr:   63
  avg:  0.288
```

Sequence of Sequences
```
- [name        , hr, avg  ]
- [Mark McGwire, 65, 0.278]
- [Sammy Sosa  , 63, 0.288]
```

Mapping of Mappings
```
Mark McGwire: {hr: 65, avg: 0.278}
Sammy Sosa: {
    hr: 63,
    avg: 0.288
  }
```

Two Documents in a Stream
```
# Ranking of 1998 home runs
---
- Mark McGwire
- Sammy Sosa
- Ken Griffey

# Team ranking
---
- Chicago Cubs
- St Louis Cardinals
```

Node for “Sammy Sosa” appears twice in this document
```
hr:
  - Mark McGwire
  # Following node labeled SS
  - &SS Sammy Sosa
rbi:
  - *SS # Subsequent occurrence
  - Ken Griffey
```

>重复的节点首先由锚点`&`识别，然后使用`*`引用。

Integers
```
canonical: 12345
decimal: +12,345
sexagesimal: 3:25:45
octal: 014
hexadecimal: 0xC
```

Floating Point
```
canonical: 1.23015e+3
exponential: 12.3015e+02
sexagesimal: 20:30.15
fixed: 1,230.15
negative infinity: -.inf
not a number: .NaN
```

Miscellaneous
```
null: ~
true: y
false: n
string: '12345'
```

Timestamps
```
canonical: 2001-12-15T02:59:43.1Z
iso8601: 2001-12-14t21:59:43.10-05:00
spaced: 2001-12-14 21:59:43.10 -5
date: 2002-12-14
```

newlines become spaces
```
---
  Mark McGwire's
  year was crippled
  by a knee injury.
```

newlines are preserved
```
--- |
  Mark McGwire's
  year was crippled
  by a knee injury.
```

Folded newlines are preserved for "more indented" and blank lines
```
txt: >
  Sammy Sosa completed another
  fine season with great stats.

    63 Home Runs
    0.288 Batting Average

  What a year!
```
