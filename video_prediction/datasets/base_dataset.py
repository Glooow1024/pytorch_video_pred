# 作者：李溢
# 日期：2019/5/6

import glob
import os
import random
import re
import h5py
from collections import OrderedDict

import numpy as np
from tensorflow.contrib.training import HParams
import torch.utils.data as data

class BaseVideoDataset(data.Dataset):
    def __init__(self, input_dir, mode='train', num_epochs=None, seed=None,
                 hparams_dict=None, hparams=None):
        ### 主要功能：读取 input_dir 文件夹下所有的 h5 文件，将文件名存为一个 list 5/19
        """
        Args:
            input_dir: either a directory containing subdirectories train,  ### 5/3
                val, test, etc, or a directory containing the tfrecords.
            mode: either train, val, or test
            num_epochs: if None, dataset is iterated indefinitely.
            seed: random seed for the op that samples subsequences.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).

        Note:
            self.input_dir is the directory containing the tfrecords.
        """
        self.input_dir = os.path.normpath(os.path.expanduser(input_dir))
        self.mode = mode
        self.num_epochs = num_epochs
        self.seed = seed

        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('Invalid mode %s' % self.mode)

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir %s does not exist" % self.input_dir)
        self.filenames = None
        # look for tfrecords in input_dir and input_dir/mode directories
        for input_dir in [self.input_dir, os.path.join(self.input_dir, self.mode)]:  ### 照应前面注释 5/3
            ### glob.glob()匹配所有的符合条件的文件，并将其以list的形式返回 5/3
            filenames = glob.glob(os.path.join(input_dir, '*.h5*'))  ### tfrecords改为h5 5/4
            if filenames:
                self.input_dir = input_dir
                self.filenames = sorted(filenames)  # ensures order is the same across systems
                break
        if not self.filenames:
            raise FileNotFoundError('No h5 were found in %s.' % self.input_dir)  ### tfrecords改为h5 5/4
        ### os.path.split(),以 "PATH" 中最后一个 '/' 作为分隔符，将“文件名”和“路径”分割开，并不智能 5/3
        ### os.path.basename()，返回文件名 5/3
        ### 这里就是根据数据集的路径来确定数据集的名称dataset_name 5/3
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])

        self.state_like_names_and_shapes = OrderedDict()   ### 这俩是干嘛的？ 5/4s
        self.action_like_names_and_shapes = OrderedDict()

        self.hparams = self.parse_hparams(hparams_dict,hparams)
        
    def __getitem__(self,index):
        ### 每个 h5 文件中是一个数据 5/19
        ### 含有 'images','speed','angle'
        ### 其中 images.sahpe = (30,160,320,3) 
        f = h5py.File(self.filenames[index], 'r')
        sample = dict(f)
        for k in sample.keys():   ### 将h5文件中的dataset数据类型转化为np类型
            sample[k] = f[k].value.astype(np.float32)
        f.close()
        return sample
    
    def __len__(self):
        return len(self.filenames)

    def get_default_hparams_dict(self):
        """
        Returns:
            A dict with the following hyperparameters.

            crop_size: crop image into a square with sides of this length.
            scale_size: resize image to this size after it has been cropped.
            context_frames: the number of ground-truth frames to pass in at
                start.
            sequence_length: the number of frames in the video sequence, so
                state-like sequences are of length sequence_length and
                action-like sequences are of length sequence_length - 1.
                This number includes the context frames.
            long_sequence_length: the number of frames for the long version.
                The default is the same as sequence_length.
            frame_skip: number of frames to skip in between outputted frames,
                so frame_skip=0 denotes no skipping.
            time_shift: shift in time by multiples of this, so time_shift=1
                denotes all possible shifts. time_shift=0 denotes no shifting.
                It is ignored (equiv. to time_shift=0) when mode != 'train'.
            force_time_shift: whether to do the shift in time regardless of
                mode.
            shuffle_on_val: whether to shuffle the samples regardless if mode
                is 'train' or 'val'. Shuffle never happens when mode is 'test'.
            use_state: whether to load and return state and actions.
        """
        hparams = dict(
            crop_size=0,
            scale_size=0,
            context_frames=1,
            sequence_length=0,
            long_sequence_length=0,
            frame_skip=0,
            time_shift=1,
            force_time_shift=False,
            shuffle_on_val=False,
            use_state=False,
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())


    def parse_hparams(self, hparams_dict, hparams):  ### 用hparams_dict和hparams更新/补充超参数 5/3
        #parsed_hparams = self.get_default_hparams_dict().update(hparams_dict or {})  ### 5/4
        parsed_hparams = self.get_default_hparams().override_from_dict(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                parsed_hparams.parse(hparam)
        if parsed_hparams.long_sequence_length == 0:
            parsed_hparams.long_sequence_length = parsed_hparams.sequence_length
        return parsed_hparams

    @property   ### 装饰器，负责把一个getter方法变成属性调用，如果不定义@.setter就是一个只读属性 5/3
    def jpeg_encoding(self):
        raise NotImplementedError

    def set_sequence_length(self, sequence_length):
        self.hparams.sequence_length = sequence_length

    #def filter(self, serialized_example):
    #    return tf.convert_to_tensor(True)  ### 把python的变量类型转换成tensor 5/3

    def parser(self, serialized_example):
        ### softmotion_dataset.py中SoftmotionVideoDataset子类定义了,下面也定义了 5/3
        """
        Parses a single tf.train.Example or tf.train.SequenceExample into
        images, states, actions, etc tensors.
        """
        raise NotImplementedError

    '''
    def make_dataset(self, batch_size):
        filenames = self.filenames   ### '.tfrecords'文件名（含路径）构成的列表 5/3
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
        ### 打乱文件名 5/3
        if shuffle:
            random.shuffle(filenames)

        ### 从'.tfrecords'文件获得序列化数据 5/3
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024)
        dataset = dataset.filter(self.filter)
        ### 打乱数据 5/3
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.num_epochs))
        else:
            dataset = dataset.repeat(self.num_epochs)

        def _parser(serialized_example):
            state_like_seqs, action_like_seqs = self.parser(serialized_example)
            seqs = OrderedDict(list(state_like_seqs.items()) + list(action_like_seqs.items()))
            return seqs

        num_parallel_calls = None if shuffle else 1  # for reproducibility (e.g. sampled subclips from the test set)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            _parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
        dataset = dataset.prefetch(batch_size)  ### 加载到缓冲通道 5/4
        return dataset
    '''

    '''
    def make_batch(self, batch_size):
        dataset = self.make_dataset(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    '''

    '''
    def decode_and_preprocess_images(self, image_buffers, image_shape):
        ### 在parser()中被调用 5/4
        ### 待修改 5/4
        def decode_and_preprocess_image(image_buffer):
            image_buffer = tf.reshape(image_buffer, [])  ### 为什么是[]?image_buffer应该是1*1的？ 5/4
            ### jpeg解码 5/4
            if self.jpeg_encoding:
                image = tf.image.decode_jpeg(image_buffer)
            else:
                image = tf.decode_raw(image_buffer, tf.uint8)
            ### 裁剪和放缩 5/4
            image = tf.reshape(image, image_shape)
            crop_size = self.hparams.crop_size
            scale_size = self.hparams.scale_size
            if crop_size or scale_size:
                if not crop_size:
                    crop_size = min(image_shape[0], image_shape[1])
                image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
                image = tf.reshape(image, [crop_size, crop_size, 3])
                if scale_size:
                    # upsample with bilinear interpolation but downsample with area interpolation
                    if crop_size < scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.BILINEAR)
                    elif crop_size > scale_size:
                        image = tf.image.resize_images(image, [scale_size, scale_size],
                                                       method=tf.image.ResizeMethod.AREA)
                    else:
                        # image remains unchanged
                        pass
            return image

        if not isinstance(image_buffers, (list, tuple)):
            image_buffers = tf.unstack(image_buffers)  ### 对矩阵进行分解的函数 5/4
        images = [decode_and_preprocess_image(image_buffer) for image_buffer in image_buffers]
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        return images
    '''

    def slice_sequences(self, state_like_seqs, action_like_seqs, example_sequence_length):
        ### 其他地方没用到？ 5/19
        """
        Slices sequences of length `example_sequence_length` into subsequences
        of length `sequence_length`. The dicts of sequences are updated
        in-place and the same dicts are returned.
        """
        # handle random shifting and frame skip
        sequence_length = self.hparams.sequence_length  # desired sequence length
        frame_skip = self.hparams.frame_skip   ### 使用的两帧之间相隔几帧 5/4
        time_shift = self.hparams.time_shift   ### 
        if (time_shift and self.mode == 'train') or self.hparams.force_time_shift:
            assert time_shift > 0 and isinstance(time_shift, int)
            #if isinstance(example_sequence_length, tf.Tensor):
            #    example_sequence_length = tf.cast(example_sequence_length, tf.int32)
            example_sequence_length = np.array(example_sequence_length).astype(np.int32)
            num_shifts = ((example_sequence_length - 1) - (sequence_length - 1) * (frame_skip + 1)) // time_shift
            assert_message = ('example_sequence_length has to be at least %d when '
                              'sequence_length=%d, frame_skip=%d.' %
                              ((sequence_length - 1) * (frame_skip + 1) + 1,
                               sequence_length, frame_skip))
            '''with tf.control_dependencies([tf.assert_greater_equal(num_shifts, 0,
                    data=[example_sequence_length, num_shifts], message=assert_message)]):
                t_start = tf.random_uniform([], 0, num_shifts + 1, dtype=tf.int32, seed=self.seed) * time_shift'''
            ### 生成一个0~num_shifts+1的随机数,不包括num_shifts+1 5/4
            t_start = np.random.randint(0, num_shifts + 1, dtype=np.int32) * time_shift
        else:
            t_start = 0
        state_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1) + 1, frame_skip + 1)
        action_like_t_slice = slice(t_start, t_start + (sequence_length - 1) * (frame_skip + 1))

        ### 按照切片取出需要的帧 5/4
        for example_name, seq in state_like_seqs.items():
            seq = seq[state_like_t_slice]
            seq = np.reshape(seq, [sequence_length] + seq.shape.as_list()[1:])
            state_like_seqs[example_name] = seq
        for example_name, seq in action_like_seqs.items():
            seq = seq[action_like_t_slice]
            seq = np.reshape(seq, [(sequence_length - 1) * (frame_skip + 1)] + list(seq.shape)[1:])
            # concatenate actions of skipped frames into single macro actions
            ### 把中间隔的几帧的action都合并到一个里边 5/4
            seq = np.reshape(seq, [sequence_length - 1, -1])
            action_like_seqs[example_name] = seq
        return state_like_seqs, action_like_seqs

    def num_examples_per_epoch(self):
        raise NotImplementedError


class VideoDataset(BaseVideoDataset):
    """
    This class supports reading tfrecords where a sequence is stored as
    multiple tf.train.Example and each of them is stored under a different
    feature name (which is indexed by the time step).
    """
    def __init__(self, *args, **kwargs):
        super(VideoDataset, self).__init__(*args, **kwargs)
        self._max_sequence_length = None
        self._dict_message = None
        
    def _check_or_infer_shapes(self):
        """
        Should be called after state_like_names_and_shapes and
        action_like_names_and_shapes have been finalized.
        """
        ### 这个函数可以丢弃？ 5/4
        raise NotImplementedError
    
    def set_sequence_length(self, sequence_length):
        if not sequence_length:
            sequence_length = (self._max_sequence_length - 1) // (self.hparams.frame_skip + 1) + 1
        self.hparams.sequence_length = sequence_length
        
    def parser(self, serialized_example):
        """
        Parses a single tf.train.Example into images, states, actions, etc tensors.
        """
        ### 可以丢弃？ 5/4
        ### self.state_like_names_and_shapes.items() 和 self.action_like_names_and_shapes.items()是什么东西？ 5/4
        ### state_like_seqs 和 action_like_seqs 又是什么东西？ 5/4
        raise NotImplementedError
        
if __name__ == '__main__':
    ### 测试用,训练时不会执行下面的代码 5/4
    import cv2
    from video_prediction import datasets

    datasets = [
        datasets.SV2PVideoDataset('data/shape', mode='val'),
        datasets.SV2PVideoDataset('data/humans', mode='val'),
        datasets.SoftmotionVideoDataset('data/bair', mode='val'),
        datasets.KTHVideoDataset('data/kth', mode='val'),
        datasets.KTHVideoDataset('data/kth_128', mode='val'),
        #datasets.UCF101VideoDataset('data/ucf101', mode='val'),  ###我删的2019/3/3
    ]
    batch_size = 4

    for dataset in datasets:
        inputs = dataset.make_batch(batch_size)
        images = inputs['images']
        images = tf.reshape(images, [-1] + images.get_shape().as_list()[2:])
        images = sess.run(images)
        images = (images * 255).astype(np.uint8)
        for image in images:
            if image.shape[-1] == 1:
                image = np.tile(image, [1, 1, 3])
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(dataset.input_dir, image)
            cv2.waitKey(50)
