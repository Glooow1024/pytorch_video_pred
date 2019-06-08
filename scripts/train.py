# liyi,209/6/8

import argparse
import json
import os
import random
import time
import numpy as np
import torch

import video_prediction.globalvar as gl
gl._init()
device = gl.get_value()

from video_prediction.datasets.base_dataset import BaseVideoDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories train, val, test, etc, or a directory containing the tfrecords")
    parser.add_argument("--val_input_dir", type=str, help="directories containing the tfrecords. default: input_dir")
    parser.add_argument("--logs_dir", default='logs', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. default is logs_dir/model_fname, where model_fname consists of information from model and model_hparams")
    parser.add_argument("--output_dir_postfix", default="")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--resume", action='store_true', help='resume from lastest checkpoint in output_dir.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")   ### 6/8
    parser.add_argument("--max_steps", type=int, default=100000, help="max steps of training")   ### 300000 6/8
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--dataset_hparams_dict", type=str, help="a json file of dataset hyperparameters")
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")
    parser.add_argument("--summary_freq", type=int, default=1000, help="save frequency of summaries (except for image and eval summaries) for train/validation set")
    parser.add_argument("--image_summary_freq", type=int, default=5000, help="save frequency of image summaries for train/validation set")
    parser.add_argument("--eval_summary_freq", type=int, default=25000, help="save frequency of eval summaries for train/validation set")
    parser.add_argument("--accum_eval_summary_freq", type=int, default=100000, help="save frequency of accumulated eval summaries for validation set only")
    parser.add_argument("--progress_freq", type=int, default=100, help="display progress every progress_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save frequence of model, 0 to disable")

    parser.add_argument("--aggregate_nccl", type=int, default=0, help="whether to use nccl or cpu for gradient aggregation in multi-gpu training")
    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    ### 设置随机数种子 5/3
    if args.seed is not None:
        torch.manual_seed(args.seed)    # 为CPU设置种子用于生成随机数，以使得结果是确定的 6/8
        torch.cuda.manual_seed(args.seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all()    # 使用多个GPU,为所有的GPU设置种子
        np.random.seed(args.seed)
        random.seed(args.seed)
        
    ### 确定输出路径 5/3
    if args.output_dir is None:
        list_depth = 0
        model_fname = ''
        for t in ('model=%s,%s' % (args.model, args.model_hparams)):
            if t == '[':
                list_depth += 1
            if t == ']':
                list_depth -= 1
            if list_depth and t == ',':
                t = '..'
            if t in '=,':
                t = '.'
            if t in '[]':
                t = ''
            model_fname += t
            
        args.output_dir = os.path.join(args.logs_dir, model_fname) + args.output_dir_postfix
        
    ### 断点训练 5/3
    if args.resume:
        if args.checkpoint:
            raise ValueError('resume and checkpoint cannot both be specified')
        args.checkpoint = args.output_dir
        
    ### 加载数据集和模型的超参数 5/3
    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.dataset_hparams_dict:
        with open(args.dataset_hparams_dict) as f:
            dataset_hparams_dict.update(json.loads(f.read()))   ### 更新字典 5/3
    if args.model_hparams_dict:
        with open(args.model_hparams_dict) as f:
            model_hparams_dict.update(json.loads(f.read()))
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
            
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        try:
            with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                dataset_hparams_dict.update(json.loads(f.read()))
        except FileNotFoundError:
            print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict.update(json.loads(f.read()))
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    ### 生成数据集 6/8
    train_dataset = BaseVideoDataset(input_dir = './data/comma', mode='train', hparams_dict=dataset_hparams_dict)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)
    dataiter = iter(train_loader)
    val_dataset = None   ### 待完成 6/8
    
    ### 确定模型 6/8
    hparams_dict = dict(model_hparams_dict)
    hparams_dict.update({
        'context_frames': train_dataset.hparams.context_frames,
        'sequence_length': train_dataset.hparams.sequence_length,
        'repeat': train_dataset.hparams.time_shift,
    })
    image_shape = None
    model = SAVPModel(image_shape, 'train', hparams_dict=hparams_dict)
    model.to(device)
    
    for step in range(0, args.max_steps):
        samples = dataiter.next()
        
        
    
    
    