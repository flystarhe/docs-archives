# Start

## install
```bash
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.10.1
pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow-gpu==1.10.1
```

## devices
```python
from tensorflow.python.client import device_lib
[(x.name, x.device_type, x.memory_limit) for x in device_lib.list_local_devices()]
## [('/device:CPU:0', 'CPU', 268435456), ('/device:GPU:0', 'GPU', 5003214848)]
```

或者:
```python
import tensorflow as tf
tf.test.is_gpu_available(), tf.test.gpu_device_name()
## (True, '/device:GPU:0')
```

## test
```python
import tensorflow as tf
sess = tf.Session()

hello = tf.constant("Hello, TensorFlow!")
print(sess.run(hello))
## "Hello, TensorFlow!"
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))
## 42

sess.close()
```
