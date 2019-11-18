import os
import time
import argparse
import json


def get_config():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # dataset
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='kmnist49')
    parser.add_argument('--batch_size', type=int, default=128)

    # training
    parser.add_argument('--n_epoch', type=int, default=14)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--model_state_path', type=str, default='')

    # misc
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--checkpoint_interval', type=int, default=500)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()

    time_str = time.strftime("%Y%m%d-%H%M%S")
    config_name = f'{time_str}_{args.dataset_name}'
    args.log_dir = os.path.join(args.log_dir, config_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, config_name)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    with open(os.path.join(args.log_dir, 'config.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    return args
