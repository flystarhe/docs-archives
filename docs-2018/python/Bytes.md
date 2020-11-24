# Bytes

## hex vs int
```python
hex(16), int("0x10", 16), int("10", 16)
```

结果:
```
('0x10', 16, 16)
```

类似有:`oct`和`bin`.

## int vs char
```python
ord("牛"), chr(29275)
```

结果:
```
(29275, '牛')
```

## int vs bytes
```python
import struct
(struct.pack("<HH", 1, 2),
struct.pack("<LL", 1, 2),
struct.unpack("<HH", b'\x01\x00\x02\x00'),
struct.unpack("<LL", b'\x01\x00\x00\x00\x02\x00\x00\x00'))
```

结果:
```
(b'\x01\x00\x02\x00', b'\x01\x00\x00\x00\x02\x00\x00\x00', (1, 2), (1, 2))
```

## to bytes
```python
("牛牛".encode("utf8"),
bytes().fromhex("e7899be7899b"),
bytes(map(ord, "\xe7\x89\x9b\xe7\x89\x9b")),
bytes([0xe7,0x89,0x9b,0xe7,0x89,0x9b]),
bytes([231, 137, 155, 231, 137, 155]))
```

结果:
```
(b'\xe7\x89\x9b\xe7\x89\x9b',
 b'\xe7\x89\x9b\xe7\x89\x9b',
 b'\xe7\x89\x9b\xe7\x89\x9b',
 b'\xe7\x89\x9b\xe7\x89\x9b',
 b'\xe7\x89\x9b\xe7\x89\x9b')
```

## to string
```python
b'\xe7\x89\x9b\xe7\x89\x9b'.decode("utf8")
```

结果:
```
'牛牛'
```
