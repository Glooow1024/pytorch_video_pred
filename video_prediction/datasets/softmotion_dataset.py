import itertools
import os
import re
import torch
from .base_dataset import VideoDataset

class SoftmotionVideoDataset(VideoDataset):
    """
    https://sites.google.com/view/sna-visual-mpc
    """
    def __init__(self, *args, **kwargs):
        super(SoftmotionVideoDataset, self).__init__(*args, **kwargs)
        
    def get_default_hparams_dict(self):
        default_hparams = super(SoftmotionVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=12,
            long_sequence_length=30,
            time_shift=2,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))
    