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
from model.model import TrajectoryModel


def options():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset_path', type=str, default='./data/CODA/')
    parser.add_argument('--dataset_name', type=str, default='coda')
    parser.add_argument('--config', type=str, default='./configs/coda.yaml')
    parser.add_argument('--single_pred_label', type=int, default=-1)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/coda_best.pth')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = options()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if args.single_pred_label >= 0:
        scale = [config['scale'][args.single_pred_label]]
    else:
        scale = config['scale']
    test_dataset = TrajectoryDataset(args.dataset_path, args.dataset_name, 'val', args.single_pred_label)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    model = TrajectoryModel(num_class=config['num_class'], in_size=2, obs_len=config['obs_len'], pred_len=config['pred_len'], embed_size=config['embed_size'], 
                            num_decode_layers=config['num_decode_layers'], scale=scale, pred_single=(args.single_pred_label>=0))
    model.load_state_dict(torch.load(args.checkpoint))
    model.cuda()
    model.eval()
    
    total_min_ade = [[0 for _ in range(len(config['test_topK']))] for _ in range(config['num_class'])]
    total_min_fde = [[0 for _ in range(len(config['test_topK']))] for _ in range(config['num_class'])]
    num_traj = [0 for _ in range(config['num_class'])]
    count = 0
    for data_batch in test_dataloader:
        data_batch = [tensor.cuda() for tensor in data_batch]
        obs, futures, neis, nei_masks, self_labels, nei_labels, refs, rot_mats = data_batch
        with torch.no_grad():
            preds, scores, _ = model(obs, neis, nei_masks, self_labels, nei_labels)
            scores = F.softmax(scores, dim=-1)
            
            min_ade = [0 for _ in range(len(config['test_topK']))]
            min_fde = [0 for _ in range(len(config['test_topK']))]
            for i, topK in enumerate(config['test_topK']):
                topK_scores, topK_indices = torch.topk(scores, topK, dim=-1) # [B topK], [B topK]
                topK_preds = torch.gather(preds, 1, topK_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, preds.size(-2), preds.size(-1))) # [B topK pred_len in_size]
                gt = futures.unsqueeze(1).repeat(1, topK, 1, 1)
                dist = torch.sqrt(torch.sum((topK_preds - gt) ** 2, dim=-1)) # [B num_init_trajs pred_len]
                ade = torch.mean(dist, dim=-1) # [B topK]
                fde = dist[:, :, -1] # [B topK]
                min_ade[i], _ = torch.min(ade, dim=1) # [B], [B]
                min_fde[i], _ = torch.min(fde, dim=1) # [B], [B]
            
            for i in range(config['num_class']):
                mask = (self_labels == i)
                num = torch.sum(mask)
                num_traj[i] += num.item()
                if num == 0:
                    continue
                for j in range(len(config['test_topK'])):
                    total_min_ade[i][j] += torch.sum(min_ade[j][mask]).item()
                    total_min_fde[i][j] += torch.sum(min_fde[j][mask]).item()
        count += 1
        if count % 100 == 0:
            print('Processed {} / {}'.format(count, len(test_dataloader)))

    for i in range(config['num_class']):
        if num_traj[i] == 0:
            continue
        for j in range(len(config['test_topK'])):
            total_min_ade[i][j] /= num_traj[i]
            total_min_fde[i][j] /= num_traj[i]
            
    print('ADE: (topK={})'.format(config['test_topK']))
    for i in range(config['num_class']):
        print('Class {}:'.format(i), end=' ')
        for j in range(len(config['test_topK'])):
            print('{:.4f}'.format(total_min_ade[i][j]), end=' ')
        print()
    print('FDE: (topK={})'.format(config['test_topK']))
    for i in range(config['num_class']):
        print('Class {}:'.format(i), end=' ')
        for j in range(len(config['test_topK'])):
            print('{:.4f}'.format(total_min_fde[i][j]), end=' ')
        print()