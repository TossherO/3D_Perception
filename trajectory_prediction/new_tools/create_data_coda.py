import os
import json
import yaml
import argparse
import numpy as np
import mmengine
from pyquaternion import Quaternion


BBOX_CLASS_REMAP = {
    "Car": "Car",
    "Service Vehicle": "Car",
    "Pickup Truck": "Car",
    "Utility Vehicle": "Car",
    "Delivery Truck": "Car",
    "Bus": "Car",
    "Pedestrian": "Pedestrian",
    "Bike": "Cyclist",
    "Scooter": "Cyclist",
    "Motorcycle": "Cyclist",
    "Skateboard": "Cyclist",
    "Segway": "Cyclist"
}

BBOX_CLASS_TO_ID = {
    "Car"                   : 0,
    "Pedestrian"            : 1,
    "Cyclist"               : 2
}


def options():
    parser = argparse.ArgumentParser(description='CODA converting ...')
    parser.add_argument('--root_path',type=str,default='./data/CODA/')
    parser.add_argument('--split',type=str,default='split2.json')
    parser.add_argument('--config',type=str,default='./configs/coda.yaml')
    args = parser.parse_args()
    return args


def data_preprocess(tracks, config):
    
    obs_len = config['obs_len']
    pred_len = config['pred_len']
    nei_radius = config['nei_radius']
    data_list = []
    all_label_ids = list(tracks.keys())
    all_labels = np.array([int(label_id.split('_')[0]) for label_id in all_label_ids])
    all_tracks = np.array([tracks[k]['data'] for k in all_label_ids])

    for i in range(len(all_tracks)):
        if all_tracks[i][-1][0] > 1e8 or all_tracks[i][obs_len-2][0] > 1e8 or all_labels[i] == 3:
            continue
        ob = all_tracks[i].copy()
        for j in range(obs_len - 2, -1, -1):
            if ob[j][0] > 1e8:
                ob[j] = ob[j+1]
        max_step = 0
        for j in range(1, obs_len + pred_len):
            max_step = max(max_step, np.sqrt(np.sum((ob[j] - ob[j-1])**2)))
        if max_step > 2:
            continue
        nei = all_tracks[np.arange(len(all_tracks)) != i, :obs_len]
        nei_labels = all_labels[np.arange(len(all_labels)) != i]
        now_nei_radius = [nei_radius[label] for label in nei_labels]
        dist_threshold = np.maximum(nei_radius[all_labels[i]], now_nei_radius)
        dist = np.linalg.norm(ob[:obs_len].reshape(1, obs_len, 2) - nei, axis=-1)
        dist = np.min(dist, axis=-1)
        nei = nei[dist < dist_threshold]
        nei_labels = nei_labels[dist < dist_threshold]
        data_list.append({'ob': ob[:obs_len], 'nei': nei, 'future': ob[obs_len:],
                            'label': all_labels[i], 'nei_label': nei_labels})
    return data_list


def update_tracks(tracks, labels, ids, xys, config):

    obs_len = config['obs_len']
    pred_len = config['pred_len']
    is_updated = {k: False for k in tracks.keys()}
    for i in range(len(labels)):
        label_id = str(labels[i]) + '_' + str(ids[i])
        if tracks.get(label_id) is None:
            tracks[label_id] = {
                'data': [[1e9, 1e9] for _ in range(obs_len + pred_len - 1)] + [xys[i].tolist()], 'label': labels[i], 
                'lost_frame': 0}
        else:
            tracks[label_id]['data'].pop(0)
            tracks[label_id]['data'].append(xys[i].tolist())
            tracks[label_id]['lost_frame'] = 0
            is_updated[label_id] = True

    for k in is_updated.keys():
        if not is_updated[k]:
            if tracks[k]['lost_frame'] < obs_len + pred_len:
                tracks[k]['data'].pop(0)
                tracks[k]['data'].append([1e9, 1e9])
                tracks[k]['lost_frame'] += 1
            else:
                del tracks[k]
    return tracks


def create_trajecotry_split(root_path, split_meta, config, split):
    save_dir = root_path + ('train' if split == 'train' else 'val')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_list = []
    for scene in split_meta[split]['scenes']:
        scene_idx = scene['scene']
        frame_list = scene['frames']
        pose_list = open(os.path.join(root_path, 'poses', 'dense', f'{scene_idx}.txt')).readlines()
        tracks = {}
        
        for frame in frame_list:
            ego2global = np.array(pose_list[frame].split()[1:]) # [x, y, z, qw, qx, qy, qz]
            ego2global_t = np.array(ego2global[:3], dtype=np.float32)
            ego2global_r = Quaternion(ego2global[3:]).rotation_matrix
            bboxes = json.load(open(os.path.join(root_path, '3d_bbox', 'os1', scene_idx, f'3d_bbox_os1_{scene_idx}_{frame}.json')))
            update_labels = []
            update_ids = []
            update_xys = []

            for bbox in bboxes['3dbbox']:
                if bbox['classId'] not in BBOX_CLASS_REMAP:
                    continue
                label = BBOX_CLASS_TO_ID[BBOX_CLASS_REMAP[bbox['classId']]]
                if label == 2 and bbox['h'] < 1.5:
                    continue
                x, y, z = bbox['cX'], bbox['cY'], bbox['cZ']
                position = np.dot(ego2global_r, np.array([x, y, z])) + ego2global_t
                id = int(bbox['instanceId'].split(':')[-1])
                update_labels.append(label)
                update_ids.append(id)
                update_xys.append(position[:2])   
            # add ego robot
            update_labels.append(3)
            update_ids.append(0)
            update_xys.append(ego2global_t[:2])

            update_tracks(tracks, update_labels, update_ids, update_xys, config)
            new_data = data_preprocess(tracks, config)
            data_list += new_data
        print(f'Scene {scene_idx} frame {frame_list[0]}_{frame_list[-1]} done!')

    save_path = os.path.join(root_path, f'coda_traj_{split}.pkl')
    mmengine.dump(data_list, save_path)
    print(f'Save {save_path} done!')


if __name__ == "__main__":
    args = options()
    root_path = args.root_path
    split = args.split
    split_path = root_path + split
    split_meta = json.load(open(split_path, 'r'))
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    create_trajecotry_split(root_path, split_meta, config, 'train')
    create_trajecotry_split(root_path, split_meta, config, 'val')