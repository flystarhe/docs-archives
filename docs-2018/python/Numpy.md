# Numpy

## 数组堆叠
`np.stack`可在不同轴上把数组堆叠在一起.而`hstack`沿水平方向(column wise)将数组拼接在一起,而`vstack`沿垂直方向(row wise)将两个数组拼接在一起,而`dstack`沿深度方向(along third axis)将数组拼接在一起.
```
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
```

`dstack`与`np.stack((mask, mask, mask), axis=2)`等价:
```
>>> mask = np.arange(16).reshape((4,4))
>>> mask = np.dstack((mask, mask, mask))
>>> mask.shape
(4, 4, 3)
```

`column_stack`可堆叠一维数组为二维数组的列:
```
>>> a = np.array([4.,2.])
>>> b = np.array([3.,8.])
>>> np.column_stack((a,b))     # returns a 2D array
array([[ 4., 3.],
       [ 2., 8.]])
>>> np.hstack((a,b))           # the result is different
array([ 4., 2., 3., 8.])
>>> np.hstack((a[:,np.newaxis],b[:,np.newaxis]))  # the result is the same
array([[ 4.,  3.],
       [ 2.,  8.]])
```

与`column_stack`相似,`row_stack`函数等价于二维数组中的`vstack`.一般在高于二维的情况中,`hstack`沿第二个维度堆叠,`vstack`沿第一个维度堆叠,而`concatenate`更进一步可以在任意给定的维度上堆叠两个数组,当然这要求其它维度的长度都相等.

- `stack(arrays, axis=0, out=None)`: Join a sequence of arrays along a new axis
- `concatenate((a1, a2, ...), axis=0, out=None)`: Join a sequence of arrays along an existing axis

## 参考资料:
- [scipy.org: Quickstart tutorial](https://www.jiqizhixin.com/articles/070101)