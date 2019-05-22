# 作者：李溢
# 日期：2019/5/6

import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from video_prediction.utils.max_sv import spectral_normed_weight
from video_prediction.layers.conv import Conv2d, Conv3d

class Dense(nn.Module):
    ### 相当于一个线性单元，units是输出的特征数 5/16
    ### inputs.shape=[batch_size,-1] 5/9
    def __init__(self, input_shape, units=1, use_spectral_norm=False, use_bias=True):
        super(Dense, self).__init__()
        self.units = units
        self.input_shape = input_shape
        self.kernel_shape = [input_shape[1], units]
        self.use_spectral_norm = use_spectral_norm
        self.use_bias = use_bias
        ### 标准差设为0.02 5/9
        self.kernel = nn.Parameter(torch.randn(self.kernel_shape, dtype=torch.float32, requires_grad=True) * 0.02)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(size=[self.units], dtype=torch.float32, requires_grad=True))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, inputs):
        if self.use_spectral_norm:
            self.kernel = spectral_normed_weight(self.kernel)
        output = torch.matmul(inputs, self.kernel)
        if self.use_bias:
            output.add_(self.bias)
        return output
        

class Encoder(nn.Module):
    ### 测试后基本没问题 5/22
    ### input_shape = (NCHW) 5/22
    ### output_shape = (N,min(2**(n_layers-1),4)),只有2个维度 5/22
    ### conv2d的in_channels是否为3存疑 5/8
    ### nef 为 encoder 的 filter 个数 5/9
    ### conv2d 要求 input_shape = NCHW
    def __init__(self, input_shape, nef=64, n_layers=3):
        super(Encoder, self).__init__()
        self.input_shape = input_shape
        self.conv = {}
        self.norm = {}
        self.conv0 = nn.Conv2d(in_channels=self.input_shape[-3], 
                               out_channels=nef, kernel_size=4, stride=2, padding=(1,1))
        def make_sequence(in_channel, i):
            out_channel = nef * min(2**i, 4)
            return [nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=out_channel, 
                        kernel_size=4, stride=2,
                        padding=(1,1)),
                      nn.InstanceNorm2d(
                        num_features=out_channel,
                        eps=1e-6)], out_channel
        
        self.model_list = nn.ModuleList()
        in_channel = nef
        for i in range(1, n_layers):
            sequence, in_channel = make_sequence(in_channel, i)
            self.model_list += sequence
        '''self.model_list = nn.ModuleList([
            nn.ModuleList([nn.Conv2d(
                        in_channels=nef,
                        out_channels=nef * min(2**i, 4), 
                        kernel_size=4, stride=2,
                        padding=(0,0,1,1)),
                      nn.InstanceNorm2d(
                        num_features=out_channel,
                        eps=1e-6)]) for i in range(1, n_layers)])'''
            
        
    def forward(self, inputs):
        ### inputs 应当是 NCHW 5/8
        outputs = {}
        output = self.conv0(inputs)
        output = F.leaky_relu(output, negative_slope=0.2)
        n = 0
        outputs['encoder_%d'%n] = output     ### for visualization 5/8
        for model in self.model_list:
            n += 1
            output = model(output)
            output = F.leaky_relu(output, negative_slope=0.2)
            outputs['encoder_%d'%n] = output
        output = F.avg_pool2d(output, output.shape[2:])
        output.squeeze_(dim=-2)  ### 对HW两个维度squeeze 5/22
        output.squeeze_(dim=-1)
        outputs['output'] = output
        return outputs
    
class ImageDiscriminator(nn.Module):
    ### input 为 NCHW 5/19
    def __init__(self, input_shape, ndf=64):
        super(ImageDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.conv0 = Conv2d(in_channels=self.input_shape[-3],
                            out_channels=ndf, kernel_size=3, stride=1, padding=(0,0,1,1),
                            use_spectral_norm=True)
        self.conv1 = Conv2d(in_channels=ndf,
                            out_channels=ndf*2, kernel_size=4, stride=2, padding=(0,0,1,1),
                            use_spectral_norm=True)
        self.conv2 = Conv2d(in_channels=ndf*2,
                            out_channels=ndf*2, kernel_size=3, stride=1, padding=(0,0,1,1),
                            use_spectral_norm=True)
        self.conv3 = Conv2d(in_channels=ndf*2,
                            out_channels=ndf*4, kernel_size=4, stride=2, padding=(0,0,1,1),
                            use_spectral_norm=True)
        self.conv4 = Conv2d(in_channels=ndf*4,
                            out_channels=ndf*4, kernel_size=3, stride=1, padding=(0,0,1,1),
                            use_spectral_norm=True)
        self.conv5 = Conv2d(in_channels=ndf*4,
                            out_channels=ndf*8, kernel_size=4, stride=2, padding=(0,0,1,1),
                            use_spectral_norm=True)
        self.conv6 = Conv2d(in_channels=ndf*8,
                            out_channels=ndf*8, kernel_size=3, stride=1, padding=(0,0,1,1),
                            use_spectral_norm=True)
        
        ### dense 的 input_shape 受到前面卷积层的影响，暂时无法确定 5/19
        #ln = torch.prod(self.input_shape[1:])   ### 去掉第一个维度后余下所有维度尺寸的乘积 5/19
        #dense_input_shape = [self.input_shape[0], ln]
        #self.dense = Dense(dense_input_shape, units=1, use_spectral_norm=True)
        self.dense = None
        
        
    def forward(self, inputs):
        ### inputs 应当是 NCHW 5/8
        outputs = {}
        output = self.conv0(inputs)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv0_0'] = output     ### for visualization 5/8
        output = self.conv1(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv0_1'] = output     ### for visualization 5/8
        output = self.conv2(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv1_0'] = output     ### for visualization 5/8
        output = self.conv3(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv1_1'] = output     ### for visualization 5/8
        output = self.conv4(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv2_0'] = output     ### for visualization 5/8
        output = self.conv5(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv2_1'] = output     ### for visualization 5/8
        output = self.conv6(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv3_0'] = output     ### for visualization 5/8
        output = output.reshape([output.shape[0],-1])   ### to [batch, -1] 5/19
        if self.dense is None:
            output_shape = output.shape
            dense_input_shape = [output_shape[0], torch.prod(output_shape[1:])]
            self.dense = Dense(dense_input_shape, units=1, use_spectral_norm=True)
        output = self.dense(output)
        outputs['output'] = output
        return outputs
    
class VideoDiscriminator(nn.Module):
    ### 原代码中要求 input_shape=TBHWC 5/9
    ### 这里设为 NCDHW 5/9
    def __init__(self, input_shape, ndf=64):
        super(VideoDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        ### conv3d的输入为 NCDHW 
        self.conv0 = Conv3d(in_channels=self.input_shape[1],
                            out_channels=ndf, kernel_size=3, stride=1, padding=(0,0,1,1,1),
                            use_spectral_norm=True)
        self.conv1 = Conv3d(in_channels=ndf,
                            out_channels=ndf*2, kernel_size=4, stride=(1,2,2), padding=(0,0,1,1,1),
                            use_spectral_norm=True)
        self.conv2 = Conv3d(in_channels=ndf*2,
                            out_channels=ndf*2, kernel_size=3, stride=1, padding=(0,0,1,1,1),
                            use_spectral_norm=True)
        self.conv3 = Conv3d(in_channels=ndf*2,
                            out_channels=ndf*4, kernel_size=4, stride=(1,2,2), padding=(0,0,1,1,1),
                            use_spectral_norm=True)
        self.conv4 = Conv3d(in_channels=ndf*4,
                            out_channels=ndf*4, kernel_size=3, stride=1, padding=(0,0,1,1,1),
                            use_spectral_norm=True)
        self.conv5 = Conv3d(in_channels=ndf*4,
                            out_channels=ndf*8, kernel_size=4, stride=2, padding=(0,0,1,1,1),
                            use_spectral_norm=True)
        self.conv6 = Conv3d(in_channels=ndf*8,
                            out_channels=ndf*8, kernel_size=3, stride=1, padding=(0,0,1,1,1),
                            use_spectral_norm=True)
        
        ### input_shape 不确定 5/19
        #from functools import reduce
        #ln = reduce(lambda x,y:x * y, self.input_shape[1:])
        #dense_input_shape = [self.input_shape[0], ln]
        #self.dense = Dense(dense_input_shape, units=1, use_spectral_norm=True)
        self.dense = None
        
        
    def forward(self, inputs):
        ### inputs 应当是 NCDHW 5/8
        outputs = {}
        output = self.conv0(inputs)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv0_0'] = output     ### for visualization 5/8
        output = self.conv1(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv0_1'] = output     ### for visualization 5/8
        output = self.conv2(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv1_0'] = output     ### for visualization 5/8
        output = self.conv3(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv1_1'] = output     ### for visualization 5/8
        output = self.conv4(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv2_0'] = output     ### for visualization 5/8
        output = self.conv5(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv2_1'] = output     ### for visualization 5/8
        output = self.conv6(output)
        output = F.leaky_relu(output, negative_slope=0.1)
        outputs['sn_conv3_0'] = output     ### for visualization 5/8
        output = output.reshape([output.shape[0],-1])   ### to [batch, -1] 5/19
        if self.dense is None:
            output_shape = output.shape
            dense_input_shape = [output_shape[0], torch.prod(output_shape[1:])]
            self.dense = Dense(dense_input_shape, units=1, use_spectral_norm=True)
        output = self.dense(output)
        outputs['output'] = output
        return outputs
    
### 编写于5/15
'''
        if hparams.use_e_rnn:
            self.dense0 = Dense(input_shape, units=hparams.nef * 4)
            if hparams.rnn == 'lstm':
                self.rnn = nn.LSTM(hidden_size=hparams.nef * 4)
            elif hparams.rnn == 'gru':
                self.rnn = nn.GRU(hidden_size=hparams.nef * 4)
            else:
                raise NotImplementedError'''   ### 暂时去掉 5/19
class Posterior(nn.Module):
    ### 测试后基本没问题 5/22
    ### input 为 DNCHW 5/19
    ### output 为 D-1,N,nz[=8] 5/21
    ### 改写自savp_model.py posterior_fn 5/15
    def __init__(self, input_shape, hparams):
        super(Posterior, self).__init__()
        self.input_shape = list(input_shape)
        self.use_e_rnn = hparams.use_e_rnn  ### 默认false 5/19
        
        self.encoder = Encoder(input_shape=[np.prod(self.input_shape[0:2])]+
                               [self.input_shape[-3]*2]+self.input_shape[-2:],
                               nef=hparams.nef, n_layers=hparams.n_layers)
        out_shape = [np.prod(self.input_shape[0:2]),
                     hparams.nef * min(4, 2**(hparams.n_layers-1))]
        self.dense1 = Dense(input_shape=out_shape, units=hparams.nz)  ### input_shape要改 5/15
        self.dense2 = Dense(input_shape=out_shape, units=hparams.nz)  ### input_shape要改 5/15
        
    def forward(self, inputs):   ### 参数千万不要忘了 self! 5/22
        ### inputs应当是 NDCHW 5/16
        outputs = {}
        inputs = torch.cat([inputs[:-1], inputs[1:]], dim=-3)  ### 将连续的两帧图片在channel维度上级联 5/16
        inputs = inputs.reshape([-1]+list(inputs.shape[-3:]))  ### 变为 NCHW 5/22
        ### 加入 action uncompleted ... 
        h = self.encoder(inputs)['output']
        if self.use_e_rnn:
            h = self.dense0(h)
            h = self.rnn(h)
        z_mu = self.dense1(h).reshape([self.input_shape[0]-1]+[self.input_shape[1]]+[-1])
        outputs['z_mu'] = z_mu
        z_log_sigma_sq = self.dense2(h).reshape([self.input_shape[0]-1]+[self.input_shape[1]]+[-1])
        z_log_sigma_sq = torch.clamp(z_log_sigma_sq, -10,10)
        outputs['z_log_sigma_sq'] = z_log_sigma_sq
        return outputs
    
    
class Prior(nn.Module):
    ### 改写自savp_model.py prior_fn 5/16
    def __init__(self, input_shape, hparams):
        super(Prior, self).__init__()
        self.hparams = hparams
        self.encoder = Encoder(input_shape, nef=hparams.nef, n_layers=hparams.n_layers)### input_shape需要根据hparmas修改 5/16
        self.dense0 = Dense(input_shape, units=hparams.nef * 4)
        if hparams.rnn == 'lstm':
            self.rnn = nn.LSTM(hidden_size=hparams.nef * 4)
        elif hparams.rnn == 'gru':
            self.rnn = nn.GRU(hidden_size=hparams.nef * 4)
        else:
            raise NotImplementedError
        self.dense1 = Dense(input_shape=hparams.nef*4, units=hparams.nz)  ### input_shape要改 5/16
        self.dense2 = Dense(input_shape=hparams.nef*4, units=hparams.nz)  ### input_shape要改 5/16
        
    def forward(inputs):
        ### inputs应当是 NCHW 5/16
        outputs = {}
        ### 将连续的两帧图片在channel维度上级联 5/16
        ### context_frams 需要根据 ... 5/16
        inputs = torch.cat([inputs[:self.hparams.context_frames - 1], inputs[1:self.hparams.context_frames]], dim=-3)  
        ### 加入 action uncompleted ... 
        h = self.encoder(inputs)
        h_zeros = torch.zeros(sizes = torch.cat(
            [[self.hparams.sequence_length - self.hparams.context_frames], h.size[1:]], axis=0))
        
        h = torch.cat([h, h_zeros], axis=0)
        h = self.dense0(h)
        h = self.rnn(h)
        z_mu = self.dense1(h)
        outputs['z_mu'] = z_mu
        z_log_sigma_sq = self.dense2(h)
        z_log_sigma_sq = torch.clamp(z_log_sigma_sq, -10,10)
        outputs['z_log_sigma_sq'] = z_log_sigma_sq
        return outputs
        
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        