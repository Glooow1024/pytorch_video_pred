"""
作者：李溢
日期：2019/5/9
"""

import torch

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