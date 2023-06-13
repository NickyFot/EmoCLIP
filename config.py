import os
from datetime import datetime
import argparse

import torch

from DataLoaders.utils import col_index

def get_config(sysv):
    parser = argparse.ArgumentParser(description='Training variables.')

    parser.add_argument('--model_basename', default='baseline')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank')
    parser.add_argument('--exp_name', default=datetime.now().strftime("%Y_%m_%d-%H%M%S"))
    parser.add_argument('--fold', default="1")
    parser.add_argument('--emo', type=int, default=0, help='Index of emotion in loo experiments')

    parser.add_argument('--pretrained', default=None, help="Path to pretrained weights")
    parser.add_argument('--fromcheckpoint', default=None, help="Path to pretrained weights")
    parser.add_argument('--finetune', action='store_true', help="finetune CLIP")
    parser.set_defaults(finetune=False)
    parser.add_argument('--text', action='store_true', help="finetune CLIP text encoder")
    parser.set_defaults(text=False)
    parser.add_argument('--visual', action='store_true', help="finetune CLIP visual encoder")
    parser.set_defaults(visual=False)

    parser.add_argument('--resample', action='store_true', help="Resample minority classes")
    parser.set_defaults(resample=False)

    parser.add_argument('--DDP', action='store_false', help="Use Distributed Data Parallel")
    parser.set_defaults(noDDP=True)

    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--batch_size', type=int, default=torch.cuda.device_count()*16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')

    parser.add_argument('--dataset_root', help="path to dataset")
    parser.add_argument('--dataset_name', default='MAFW')

    # input video options
    parser.add_argument('--input_shape', nargs='+', default=224)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--clip_len', type=int, default=32)

    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=False)

    args, _ = parser.parse_known_args(sysv)
    return args
