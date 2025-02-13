import os
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))

import argparse
import time
import random
import yaml
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset


def options():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset_path', type=str, default='./data/CODA/')
    parser.add_argument('--dataset_name', type=str, default='coda')
    parser.add_argument('--config', type=str, default='./configs/coda.yaml')
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = options()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    test_dataset = TrajectoryDataset(args.dataset_path, args.dataset_name, 'val')
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    
    total_ade = [0 for _ in range(config['num_class'])]
    total_fde = [0 for _ in range(config['num_class'])]
    num_traj = [0 for _ in range(config['num_class'])]
    count = 0
    for data_batch in test_dataloader:
        data_batch = [tensor.cuda() for tensor in data_batch]
        obs, futures, neis, nei_masks, self_labels, nei_labels, refs, rot_mats = data_batch
        with torch.no_grad():
            v = obs[:, -1, :] - obs[:, -2, :]
            preds = torch.zeros_like(futures)
            for i in range(config['pred_len']):
                preds[:, i, :] = obs[:, -1, :] + v * (i + 1)
            dist = torch.sqrt(torch.sum((preds - futures) ** 2, dim=-1))
            ade = torch.mean(dist, dim=-1)
            fde = dist[:, -1]
            for i in range(config['num_class']):
                mask = self_labels == i
                num = torch.sum(mask)
                num_traj[i] += num.item()
                if num == 0:
                    continue
                total_ade[i] += torch.sum(ade[mask]).item()
                total_fde[i] += torch.sum(fde[mask]).item()
        count += 1
        if count % 100 == 0:
            print('Processed {} / {}'.format(count, len(test_dataloader)))

    for i in range(config['num_class']):
        if num_traj[i] == 0:
            continue
        total_ade[i] /= num_traj[i]
        total_fde[i] /= num_traj[i]
            
    print('ADE: ', total_ade)
    print('FDE: ', total_fde)