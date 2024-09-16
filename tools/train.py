#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import sys

# Save the command-line arguments to command.txt
start_command = ' '.join(sys.argv)

import argparse
import copy

import torch
from loguru import logger

from damo.apis import Trainer
from damo.config.base import parse_config
from damo.utils import synchronize

import os

def make_parser():
    """
    Create a parser with some common arguments used by users.

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Damo-Yolo train parser')

    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='plz input your config file',
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tea_config', type=str, default=None)
    parser.add_argument('--tea_ckpt', type=str, default=None)
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
    
    synchronize()
    if args.tea_config is not None:
        tea_config = parse_config(args.tea_config)
    else:
        tea_config = None

    config = parse_config(args.config_file)
    config.merge(args.opts)

    trainer = Trainer(config, args, tea_config,start_command=start_command)
    trainer.train(local_rank, device)

if __name__ == '__main__':
    main()
