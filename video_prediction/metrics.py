### liyi, 2019/6/8
### 待完成。。。

import torch
import numpy as np
#import lpips_tf


def mse(a, b):
    return torch.mean(torch.pow(a-b, 2), (-3, -2, -1))

def psnr(a, b):
    rmse = torch.sqrt(torch.mean(torch.pow(a-b, 2), (-3, -2, -1)))
    psnr = 20 * torch.log(255.0 / rmse)   ### 255 还是 1.。。 6/8
    return psnr


def ssim(a, b):
    return tf.image.ssim(a, b, 1.0)


def lpips(input0, input1):
    if input0.shape[-1].value == 1:
        input0 = tf.tile(input0, [1] * (input0.shape.ndims - 1) + [3])
    if input1.shape[-1].value == 1:
        input1 = tf.tile(input1, [1] * (input1.shape.ndims - 1) + [3])

    distance = lpips_tf.lpips(input0, input1)
    return -distance
