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
from new_model.model import TrajectoryModel


def options():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset_path', type=str, default='./data/CODA/')
    parser.add_argument('--dataset_name', type=str, default='coda')
    parser.add_argument('--config', type=str, default='./configs/coda.yaml')
    parser.add_argument('--lr_scaling', action='store_true', default=False)
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = options()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    train_dataset = TrajectoryDataset(args.dataset_path, args.dataset_name, 'train')
    val_dataset = TrajectoryDataset(args.dataset_path, args.dataset_name, 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn)
    model = TrajectoryModel(num_class=config['num_class'], in_size=2, obs_len=config['obs_len'], pred_len=config['pred_len'], 
                            embed_size=config['embed_size'], num_decode_layers=config['num_decode_layers'], num_modes=config['num_modes'])
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    reg_criterion = torch.nn.SmoothL1Loss().cuda()
    cls_criterion = torch.nn.BCELoss().cuda()
    if args.lr_scaling:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.6)

    global_min_ade = [100 for _ in range(config['num_class'])]
    global_min_fde = [100 for _ in range(config['num_class'])]
    for epoch in range(config['epoch']):

        model.train()
        total_loss = [0 for _ in range(config['num_class'])]
        num_traj = [0 for _ in range(config['num_class'])]
        for data_batch in train_dataloader:
            data_batch = [tensor.cuda() for tensor in data_batch]
            obs, futures, neis, nei_masks, self_labels, nei_labels, refs, rot_mats = data_batch
            preds, scores = model(obs, neis, nei_masks, self_labels, nei_labels)
            scores = F.softmax(scores, dim=-1)
            # 计算与真值平均误差最小的预测值
            gt = futures.unsqueeze(1).repeat(1, config['num_modes'], 1, 1)
            dist = torch.sqrt(torch.sum((preds - gt) ** 2, dim=-1)) # [B num_modes pred_len]
            ade = torch.mean(dist, dim=-1) # [B num_modes]
            min_ade, min_idx = torch.min(ade, dim=1) # [B], [B]
            best_preds = preds[torch.arange(preds.size(0)), min_idx] # [B pred_len in_size]
            best_scores = scores[torch.arange(scores.size(0)), min_idx] # [B]
            # 计算损失
            loss = 0
            for i in range(config['num_class']):
                mask = (self_labels == i)
                num = torch.sum(mask)
                num_traj[i] += num
                if num == 0:
                    continue
                reg_loss = reg_criterion(best_preds[mask].reshape(num, -1), futures[mask].reshape(num, -1))
                cls_loss = cls_criterion(best_scores[mask], torch.ones(num).cuda())
                loss += reg_loss + cls_loss
                total_loss[i] += (reg_loss + cls_loss).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, [total_loss[i] / num_traj[i] for i in range(config['num_class'])]))
        
        model.eval()
        total_min_ade = [0 for _ in range(config['num_class'])]
        total_min_fde = [0 for _ in range(config['num_class'])]
        num_traj = [0 for _ in range(config['num_class'])]
        for data_batch in val_dataloader:
            data_batch = [tensor.cuda() for tensor in data_batch]
            obs, futures, neis, nei_masks, self_labels, nei_labels, refs, rot_mats = data_batch
            with torch.no_grad():
                preds, scores = model(obs, neis, nei_masks, self_labels, nei_labels)
                scores = F.softmax(scores, dim=-1)
                topK_scores, topK_indices = torch.topk(scores, config['val_topK'], dim=-1) # [B topK], [B topK]
                topK_preds = torch.gather(preds, 1, topK_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, preds.size(-2), preds.size(-1))) # [B topK pred_len in_size]
                gt = futures.unsqueeze(1).repeat(1, config['val_topK'], 1, 1)
                dist = torch.sqrt(torch.sum((topK_preds - gt) ** 2, dim=-1)) # [B num_modes pred_len]
                ade = torch.mean(dist, dim=-1) # [B topK]
                fde = dist[:, :, -1] # [B topK]
                min_ade, min_ade_idx = torch.min(ade, dim=1) # [B], [B]
                min_fde, min_fde_idx = torch.min(fde, dim=1) # [B], [B]
                for i in range(config['num_class']):
                    mask = (self_labels == i)
                    num = torch.sum(mask)
                    num_traj[i] += num
                    if num == 0:
                        continue
                    total_min_ade[i] += torch.sum(min_ade[mask]).item()
                    total_min_fde[i] += torch.sum(min_fde[mask]).item()
        for i in range(config['num_class']):
            total_min_ade[i] /= num_traj[i]
            total_min_fde[i] /= num_traj[i]
        print('Val: ADE: {}, FDE: {}; Best ADE: {}, FDE: {}'.format(total_min_ade, total_min_fde, global_min_ade, global_min_fde))
        
        if args.lr_scaling:
            scheduler.step()

        if sum(total_min_ade) + sum(total_min_fde) < sum(global_min_ade) + sum(global_min_fde):
            global_min_ade = total_min_ade
            global_min_fde = total_min_fde
            torch.save(model.state_dict(), args.checkpoint + args.dataset_name + '_best.pth')
            print('New best model saved.')