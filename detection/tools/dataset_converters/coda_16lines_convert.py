import os
import numpy as np
import mmengine
import random

cam = ['cam0', 'cam1']
for split in ['train', 'val', 'test']:
    info_path = './data/CODA/coda_infos_{}.pkl'.format(split)
    if not os.path.exists(info_path):
        continue
    info = mmengine.load(info_path)
    
    for data in info['data_list']:
        data['lidar_points']['lidar_path'] = data['lidar_points']['lidar_path'].replace('3d_comp/', '3d_comp_downsample/')
        data['images'].pop(cam[random.randint(0, 1)])
        
        instances = []
        for instance in data['instances']:
            if instance['bbox_label_3d'] == 0:
                continue
            instance['bbox_label_3d'] -= 1
            instances.append(instance)
        data['instances'] = instances
        
    save_path = './data/CODA/coda_16lines_infos_{}.pkl'.format(split)
    mmengine.dump(info, save_path)