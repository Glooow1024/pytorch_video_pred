#globalvar.py
#!/usr/bin/python
# -*- coding: utf-8 -*-
# 设置全局 GPU
# liyi, 2019/6/8

import torch

def _init():
    global _device
    _device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def set_value(value):
    _device = value

def get_value(defValue=None):
    try:
        return _device
    except KeyError:
        return defValue