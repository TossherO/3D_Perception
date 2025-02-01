import os
import numpy as np
import mmengine
import random

cam = ['cam0', 'cam1']
point_cloud_range = [0, -9.6, -1.5, 14.4, 9.6, 4.5]

for split in ['train', 'val', 'test']:
    info_path = './data/CODA/coda_infos_{}.pkl'.format(split)
    if not os.path.exists(info_path):
        continue
    info = mmengine.load(info_path)
    
    new_data_list = []
    for data in info['data_list']:
        data['lidar_points']['lidar_path'] = data['lidar_points']['lidar_path'].replace('3d_comp/', '3d_comp_downsample/')
        data['images'].pop(cam[random.randint(0, 1)])
        
        instances = []
        for instance in data['instances']:
            if instance['bbox_label_3d'] == 0:
                continue
            x, y, z = instance['bbox_3d'][:3]
            if x < point_cloud_range[0] or x > point_cloud_range[3] or y < point_cloud_range[1] or y > point_cloud_range[4] or z < point_cloud_range[2] or z > point_cloud_range[5]:
                continue
            instance['bbox_label_3d'] -= 1
            instances.append(instance)
        data['instances'] = instances
        
        if len(data['instances']) > 0:
            new_data_list.append(data)
            
    info['data_list'] = new_data_list
    save_path = './data/CODA/coda_16lines_infos_{}.pkl'.format(split)
    mmengine.dump(info, save_path)