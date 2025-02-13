import os
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))
import yaml
import time
import numpy as np
import torch
import mmengine
from mmcv.ops.diff_iou_rotated import box2corners, oriented_box_intersection_2d
from tools.dataset import TrajectoryDataset
from model.model import TrajectoryModel


def iou_rotated_2d(box3d1, box3d2):
    box1 = box3d1[..., [0, 1, 3, 4, 6]]  # 2d box
    box2 = box3d2[..., [0, 1, 3, 4, 6]]
    box1 = box1.unsqueeze(1).repeat(1, box2.shape[0], 1)
    box2 = box2.unsqueeze(0).repeat(box1.shape[0], 1, 1)
    corners1 = box2corners(box1)
    corners2 = box2corners(box2)
    intersection, _ = oriented_box_intersection_2d(corners1, corners2)
    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    iou = intersection / (area1 + area2 - intersection)
    return iou


def data_preprocess(tracks, config):
    
    obs_len = config['obs_len']
    nei_radius = config['nei_radius']
    obs = []
    neis_ = []
    num_neis = []
    self_labels = []
    nei_labels_ = []
    refs = []
    rot_mats = []
    gt_ids = []
    all_label_ids = list(tracks.keys())
    all_labels = np.array([int(label_id.split('_')[0]) for label_id in all_label_ids])
    all_tracks = np.array([tracks[k]['data'] for k in all_label_ids])
    all_gt_ids = np.array([tracks[k]['gt_id'] for k in all_label_ids])

    for i in range(len(all_tracks)):
        if all_tracks[i][-1][0] > 1e8 or all_tracks[i][-2][0] > 1e8 or all_labels[i] == 3:
            continue
        ob = all_tracks[i].copy()
        for j in range(obs_len - 2, -1, -1):
            if ob[j][0] > 1e8:
                ob[j] = ob[j+1]
        nei = all_tracks[np.arange(len(all_tracks)) != i]
        nei_labels = all_labels[np.arange(len(all_labels)) != i]
        now_nei_radius = [nei_radius[label] for label in nei_labels]
        dist_threshold = np.maximum(nei_radius[all_labels[i]], now_nei_radius)
        dist = np.linalg.norm(ob[:obs_len].reshape(1, obs_len, 2) - nei, axis=-1)
        dist = np.min(dist, axis=-1)
        nei = nei[dist < dist_threshold]
        nei_labels = nei_labels[dist < dist_threshold]
        
        ref = ob[-1]
        ob = ob - ref
        if nei.shape[0] != 0:
            nei = nei - ref
        angle = np.arctan2(ob[0][1], ob[0][0])
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], 
                            [np.sin(angle), np.cos(angle)]])
        ob = np.dot(ob, rot_mat)
        if nei.shape[0] != 0:
            nei = np.dot(nei, rot_mat)
        
        obs.append(ob)
        neis_.append(nei)
        num_neis.append(nei.shape[0])
        self_labels.append(all_labels[i])
        nei_labels_.append(nei_labels)
        refs.append(ref.flatten())
        rot_mats.append(rot_mat)
        gt_ids.append(all_gt_ids[i])
        
    if len(obs) == 0:
        return None
            
    max_num_nei = max(num_neis)
    if max_num_nei == 0:
        max_num_nei = 1
    nei_masks = torch.zeros(len(obs), max_num_nei, dtype=torch.bool)
    neis = torch.zeros(len(obs), max_num_nei, obs_len, 2)
    nei_labels = torch.zeros(len(obs), max_num_nei, dtype=torch.int32) - 1
    for i in range(len(obs)):
        nei_masks[i, :num_neis[i]] = True
        neis[i, :num_neis[i]] = torch.tensor(neis_[i])
        nei_labels[i, :num_neis[i]] = torch.tensor(nei_labels_[i])
    
    obs = torch.tensor(np.stack(obs, axis=0), dtype=torch.float32).cuda()
    self_labels = torch.tensor(self_labels, dtype=torch.int32).cuda()
    refs = torch.tensor(np.stack(refs, axis=0), dtype=torch.float32).cuda()
    rot_mats = torch.tensor(np.stack(rot_mats, axis=0), dtype=torch.float32).cuda()
    neis = neis.cuda()
    nei_masks = nei_masks.cuda()
    nei_labels = nei_labels.cuda()
    return obs, neis, nei_masks, self_labels, nei_labels, refs, rot_mats, gt_ids


def update_tracks(tracks, labels, ids, xys, gt_ids, length):

    is_updated = {k: False for k in tracks.keys()}
    for i in range(len(labels)):
        label_id = str(labels[i]) + '_' + str(ids[i])
        if tracks.get(label_id) is None:
            tracks[label_id] = {
                'data': [[1e9, 1e9] for _ in range(length - 1)] + [xys[i].tolist()],
                'label': labels[i], 
                'lost_frame': 0}
        else:
            tracks[label_id]['data'].pop(0)
            tracks[label_id]['data'].append(xys[i].tolist())
            tracks[label_id]['lost_frame'] = 0
            is_updated[label_id] = True
        if gt_ids is not None:
            tracks[label_id]['gt_id'] = gt_ids[i]

    for k in is_updated.keys():
        if not is_updated[k]:
            if tracks[k]['lost_frame'] < length:
                tracks[k]['data'].pop(0)
                tracks[k]['data'].append([1e9, 1e9])
                tracks[k]['lost_frame'] += 1
            else:
                del tracks[k]
    return tracks


track_results = mmengine.load('./data/CODA/coda_track_results_full.pkl')
config_path = './configs/coda.yaml'
checkpoint = './checkpoint/coda_best.pth'
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
traj_pred_model = TrajectoryModel(num_class=config['num_class'], in_size=2, obs_len=config['obs_len'], pred_len=config['pred_len'], 
                embed_size=config['embed_size'], num_decode_layers=config['num_decode_layers'], scale=config['scale'], pred_single=False)
traj_pred_model.load_state_dict(torch.load(checkpoint))
traj_pred_model.cuda().eval()

iou_shreshold = [0.7, 0.5, 0.5]
total_min_ade = [0, 0, 0]
total_min_fde = [0, 0, 0]
num_traj = [0, 0, 0]
for result in track_results:
    tracks = {}
    gts = {}
    sequence_len = len(result['data_list'][0])
    for i in range(sequence_len):
        if i >= 24:
            labels, ids, xys, gt_ids = [], [], [], []
            for label in range(3):
                frame = result['data_list'][label][i-24]
                gt_bboxes = torch.tensor(np.array(frame['gt_bboxes']), dtype=torch.float).cuda()
                _gt_ids = [int(gt_id.split(':')[-1]) for gt_id in frame['gt_ids']]
                track_bboxes = torch.tensor(np.array(frame['track_bboxes']), dtype=torch.float).cuda()
                track_ids = np.array(frame['track_ids'])
                track_states = frame['track_states']
                valid = np.zeros(len(track_bboxes), dtype=bool)
                for k in range(len(track_bboxes)):
                    tokens = track_states[k].split('_')
                    if tokens[0] == 'birth':
                        valid[k] =  True
                    elif len(tokens) < 3:
                        valid[k] =  False
                    elif tokens[0] == 'alive' and int(tokens[1]) == 1:
                        valid[k] =  True
                track_bboxes = track_bboxes[valid]
                track_ids = track_ids[valid]
                track_gt_ids = np.zeros(len(track_bboxes), dtype=int) - 1
                
                if len(gt_bboxes) > 0 and len(track_bboxes) > 0:
                    iou_matrix = iou_rotated_2d(track_bboxes, gt_bboxes)
                    for k in range(len(track_bboxes)):
                        max_iou, max_idx = torch.max(iou_matrix[k], 0)
                        if max_iou > iou_shreshold[label]:
                            iou_matrix[:, max_idx] = 0
                            track_gt_ids[k] = _gt_ids[max_idx]
                        else:
                            track_gt_ids[k] = -1
                
                for k in range(len(track_bboxes)):
                    labels.append(label)
                    ids.append(track_ids[k])
                    xys.append(track_bboxes[k, :2].cpu().numpy())
                    gt_ids.append(track_gt_ids[k])
                
            update_tracks(tracks, labels, ids, xys, gt_ids, 16)
                
        labels, ids, xys = [], [], []
        for label in range(3):
            frame = result['data_list'][label][i]
            labels.extend([label] * len(frame['gt_bboxes']))
            ids.extend([int(gt_id.split(':')[-1]) for gt_id in frame['gt_ids']])
            xys.extend([bbox[:2] for bbox in frame['gt_bboxes']])
        update_tracks(gts, labels, ids, xys, None, 24)
        
        topK = 5
        data_input = data_preprocess(tracks, config)
        if data_input is not None:
            with torch.no_grad():
                obs, neis, nei_masks, self_labels, nei_labels, refs, rot_mats, gt_ids = data_input
                preds, scores, _ = traj_pred_model(obs, neis, nei_masks, self_labels, nei_labels)
                scores = torch.nn.functional.softmax(scores, dim=-1)
                topK_scores, topK_indices = torch.topk(scores, topK, dim=-1) # [B topK], [B topK]
                topK_preds = torch.gather(preds, 1, topK_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, preds.size(-2), preds.size(-1))) # [B topK pred_len in_size]
                rot_mats_T = rot_mats.transpose(1, 2)
                preds_ori = torch.matmul(topK_preds, rot_mats_T.unsqueeze(1)) + refs.unsqueeze(1).unsqueeze(2)
                
                for j in range(len(preds_ori)):
                    gt_id = str(self_labels[j].item()) + '_' + str(gt_ids[j])
                    if gt_ids[j] != -1 and gts.get(gt_id) is not None:
                        gt = torch.tensor(gts[gt_id]['data']).cuda().unsqueeze(0).repeat(topK, 1, 1)
                        if torch.sum(gt[:, :, 0] > 1e8) > 0:
                            continue
                        dist = torch.sqrt(torch.sum((preds_ori[j] - gt) ** 2, dim=-1))
                        if self_labels[j] == 0:
                            dist -= 1
                            dist[dist < 0] = 0
                        ade = torch.mean(dist, dim=-1)
                        fde = dist[:, -1]
                        min_ade, _ = torch.min(ade, dim=0)
                        min_fde, _ = torch.min(fde, dim=0)
                        total_min_ade[self_labels[j]] += min_ade.item()
                        total_min_fde[self_labels[j]] += min_fde.item()
                        num_traj[self_labels[j]] += 1
                        
    print('Processed {}'.format(result['token']))
                        
for i in range(3):
    if num_traj[i] == 0:
        continue
    total_min_ade[i] /= num_traj[i]
    total_min_fde[i] /= num_traj[i]
    print('Class {}:'.format(i), end=' ')
    print('{:.4f}'.format(total_min_ade[i]), end=' ')
    print('{:.4f}'.format(total_min_fde[i]))
    print('Number of trajectories: {}'.format(num_traj[i]))