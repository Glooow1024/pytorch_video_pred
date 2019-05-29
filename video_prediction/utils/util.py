"""
作者：李溢
日期：2019/5/9
"""

import torch
from tensorflow import broadcast_static_shape

def maybe_pad_or_slice(tensor, desired_length):
    ### 将 tensor 第 0 个维度调整至 desired_length 5/23
    length = list(tensor.shape)[0]
    if length < desired_length:
        paddings = torch.zeros(size=[desired_length - length]+list(tensor.shape[1:]))
        tensor = torch.cat((tensor, paddings), dim=0)
    elif length > desired_length:
        tensor = tensor[:desired_length]
    assert tensor.shape[0] == desired_length
    return tensor


def with_flat_batch(flat_batch_fn, ndims=4):
    ### 相当于一个装饰器 5/9
    ### 用 flat_batch_fn 装饰 fn 5/9
    def fn(x, *args, **kwargs):
        ### 如果 x 为图像，则应该为 NDHWC 5/9
        shape = x.shape
        flat_batch_shape = [-1]+list(shape[-(ndims-1):])
        flat_batch_x = tf.reshape(x, flat_batch_shape)  ### shape= -1HWC 5/9
        flat_batch_r = flat_batch_fn(flat_batch_x, *args, **kwargs)
        ### 对flat_batch_fn的输出的每个元素进行 rehsape 5/9
        ### reshape = NDxx
        ### 其中后面的几个维度 xx 由 flat_batch_fn 的输出决定 5/9
        r = nest.map_structure(lambda x: tf.reshape(x, tf.concat([shape[:-(ndims-1)], tf.shape(x)[1:]], axis=0)),
                               flat_batch_r)
        return r
    return fn

### 5/29
def tile_concat(values, axis):
    ### values为一个list，里边有多个tensor 5/26
    ### 实现将多个tensor通过broadcasting机制，将他们tile到shape完全相同
    ### 除了axis维度，可以保持不相同 5/26
    shapes = [v.shape() for v in values]
    # convert axis to positive form
    ndims = shapes[0].ndims
    for shape in shapes[1:]:
        assert ndims == shape.ndims
    if -ndims < axis < 0:
        axis += ndims
    # remove axis dimension
    shapes = [list(s) for s in shapes]
    dims = [shape.pop(axis) for shape in shapes]   ### pop 会弹出/返回 axis 位置处的值, shape也会改变 5/26
    shapes = [tf.TensorShape(shape) for shape in shapes]
    # compute broadcasted shape
    b_shape = shapes[0]
    for shape in shapes[1:]:
        b_shape = broadcast_static_shape(b_shape, shape)
    # add back axis dimension
    b_shapes = [list(b_shape) for _ in dims]
    for b_shape, dim in zip(b_shapes, dims):
        b_shape.insert(axis, dim)
    # tile values to match broadcasted shape, if necessary
    b_values = []
    for value, b_shape in zip(values, b_shapes):
        multiples = []
        for dim, b_dim in zip(list(value.shape), b_shape):
            if dim == b_dim:
                multiples.append(1)
            else:
                assert dim == 1
                multiples.append(b_dim)
        if any(multiple != 1 for multiple in multiples):
            b_value = torch.repeat(value, multiples)
        else:
            b_value = value
        b_values.append(b_value)
    return torch.cat(b_values, axis=axis)