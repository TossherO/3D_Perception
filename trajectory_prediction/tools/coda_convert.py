import os
import json
import argparse
import numpy as np
from pyquaternion import Quaternion


def options():
    parser = argparse.ArgumentParser(description='CODA converting ...')
    parser.add_argument('--root_path',type=str,default='./data/CODA/')
    parser.add_argument('--split',type=str,default='split2.json')
    args = parser.parse_args()
    return args


def create_trajecotry_split(root_path, split_meta, split):
    save_dir = root_path + ('train' if split == 'train' else 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for scene in split_meta[split]['scenes']:
        scene_idx = scene['scene']
        frame_list = scene['frames']
        pose_list = open(os.path.join(root_path, 'poses', 'dense', f'{scene_idx}.txt')).readlines()
        data_list = []

        for frame in frame_list:
            ego2global = np.array(pose_list[frame].split()[1:]) # [x, y, z, qw, qx, qy, qz]
            ego2global_t = np.array(ego2global[:3], dtype=np.float32)
            ego2global_r = Quaternion(ego2global[3:]).rotation_matrix
            bboxes = json.load(open(os.path.join(root_path, '3d_bbox', 'os1', scene_idx, f'3d_bbox_os1_{scene_idx}_{frame}.json')))
            
            for bbox in bboxes['3dbbox']:
                if bbox['classId'] != 'Pedestrian':
                    continue
                x, y, z = bbox['cX'], bbox['cY'], bbox['cZ']
                position = np.dot(ego2global_r, np.array([x, y, z])) + ego2global_t
                id = int(bbox['instanceId'].split(':')[-1])
                data_list.append([frame, id, position[0], position[1]])
            
        if split == 'val':
            split = 'test'
        save_path = os.path.join(root_path, split, f'{scene_idx}_{frame_list[0]}_{frame_list[-1]}.txt')
        np.savetxt(save_path, data_list, fmt='%d %d %f %f')
        print(f'Save {save_path} done!')
    # path = path_root +'anno/'
    # all_files = [str(file)+'.json' for file in load_dict[split]]
    # for file in all_files:
    #     file_group_path = ''.join([path, file])
    #     with open(file_group_path, 'r') as load_f:
    #         load_dict = json.load(load_f)
        
    #     if split == 'val':
    #         split = 'test'
    #     Path(path_root + split).mkdir(parents=True, exist_ok=True)
    #     save_path = path_root + split + '/' + file.split('.')[0] + '.txt'
    #     data_list = []
    #     for frame in load_dict['frames']:
    #         frame_id = frame['frameId']
    #         for item in frame['items']:
    #             if item['boundingbox']['z'] < 1.2:
    #                 continue
    #             id = item['number']
    #             x, y = item['position']['x'], item['position']['y']
    #             data_list.append([frame_id, id, x, y])

    #     np.savetxt(save_path, data_list, fmt='%d %d %f %f')


if __name__ == "__main__":
    args = options()
    root_path = args.root_path
    split = args.split
    split_path = root_path + split
    split_meta = json.load(open(split_path, 'r'))
    create_trajecotry_split(root_path, split_meta, 'train')
    create_trajecotry_split(root_path, split_meta, 'val')