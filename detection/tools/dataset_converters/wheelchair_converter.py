import os
import json
import numpy as np
import cv2
from pyquaternion import Quaternion
import mmengine


BBOX_CLASS_TO_ID = {
    "Pedestrian": 0,
    "Cyclist": 1
}


def create_wheelchair_infos_split(root_path,
                                info_prefix,
                                out_dir,
                                split):
    
    metainfo = {
        'categories': BBOX_CLASS_TO_ID,
        'dataset': 'wheelchair',
        'version': 'v1.0',
        'split': split}
    
    calib0 = json.load(open(os.path.join(root_path, 'calib.json')))[0]
    K = np.array([[calib0['camera_internal']['fx'], 0, calib0['camera_internal']['cx']],
                    [0, calib0['camera_internal']['fy'], calib0['camera_internal']['cy']],
                    [0, 0, 1]])
    dist = np.array([calib0['distortionK'][0], calib0['distortionK'][1], calib0['distortionP'][0], calib0['distortionP'][1]])
    cam2img, _ = cv2.getOptimalNewCameraMatrix(K, dist, (calib0['width'], calib0['height']), 1, (calib0['width'], calib0['height']))
    lidar2cam = np.array(calib0['camera_external']).reshape(4, 4)
    _cam2img = np.eye(4)
    _cam2img[:3, :3] = cam2img
    lidar2img = _cam2img @ lidar2cam
                    
    labels = json.load(open(os.path.join(root_path, f'{split}_labels.json')))
    data_list = []
    count = 0
    total = sum([sequence['length'] for sequence in labels])
    print(f'Processing {split} split with {total} samples')
    
    for sequence in labels:
        scene_idx = int(sequence['token'].replace('sequence', '').split('_')[0])
        sequence_idx = int(sequence['token'].replace('sequence', '').split('_')[1])
            
        for frame in sequence['frames']:
            data = {}
            data['scene'] = scene_idx
            data['sequence'] = sequence_idx
            data['frame'] = frame['idx']
            data['sample_idx'] = count
            data['token'] = f"{scene_idx}_{frame['idx']}"
            data['lidar_points'] = {
                'num_pts_feats': 4,
                'lidar_path': os.path.join(f'scene{scene_idx}', 'lidar_point_cloud_0', '%04d.bin' % frame['idx'])}
            
            cam0 = {}
            cam0['img_path'] = os.path.join(f'scene{scene_idx}', 'camera_image_0', '%04d.png' % frame['idx'])
            cam0['cam2img'] = cam2img
            cam0['lidar2cam'] = lidar2cam
            # cam0['lidar2img'] = lidar2img
            data['images'] = {'cam0': cam0}
            
            instances = []
            for instance in frame['instances']:
                if instance['label'] not in BBOX_CLASS_TO_ID:
                    continue
                _instance = {}
                _instance['bbox_label_3d'] = BBOX_CLASS_TO_ID[instance['label']]
                _instance['bbox_3d'] = instance['bbox']
                _instance['instance_id'] = instance['id']
                _instance['is_occluded'] = None
                instances.append(_instance)
            data['instances'] = instances
            
            data_list.append(data)
            count += 1
            if count % 100 == 0:
                print(f'{count}/{total} samples processed')
    
    info_path = os.path.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
    mmengine.dump({'metainfo': metainfo, 'data_list': data_list}, info_path)
            

def create_wheelchair_infos(root_path,
                            info_prefix,
                            out_dir,
                            version='v1.0-trainval'):
    """Create info file of coda dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        out_dir (str): Output directory of the info filenames.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
    """
    if version == 'v1.0-trainval':
        create_wheelchair_infos_split(root_path, info_prefix, out_dir, split='train')
        create_wheelchair_infos_split(root_path, info_prefix, out_dir, split='val')
    elif version == 'v1.0-train':
        create_wheelchair_infos_split(root_path, info_prefix, out_dir, split='train')
    elif version == 'v1.0-val':
        create_wheelchair_infos_split(root_path, info_prefix, out_dir, split='val')
    elif version == 'v1.0-test':
        create_wheelchair_infos_split(root_path, info_prefix, out_dir, split='test')