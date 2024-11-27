import os
import json
import yaml
import numpy as np
from pyquaternion import Quaternion
import mmengine


# BBOX_CLASS_TO_ID = {
#     # Dynamic Classes
#     "Car"                   : 0,
#     "Pedestrian"            : 1,
#     "Bike"                  : 2,
#     "Motorcycle"            : 3,
#     "Golf Cart"             : 4, # Unused
#     "Truck"                 : 5, # Unused
#     "Scooter"               : 6,
#     # Static Classes
#     "Tree"                  : 7,
#     "Traffic Sign"          : 8,
#     "Canopy"                : 9,
#     "Traffic Light"         : 10,
#     "Bike Rack"             : 11,
#     "Bollard"               : 12,
#     "Construction Barrier"  : 13, # Unused
#     "Parking Kiosk"         : 14,
#     "Mailbox"               : 15,
#     "Fire Hydrant"          : 16,
#     # Static Class Mixed
#     "Freestanding Plant"    : 17,
#     "Pole"                  : 18,
#     "Informational Sign"    : 19,
#     "Door"                  : 20,
#     "Fence"                 : 21,
#     "Railing"               : 22,
#     "Cone"                  : 23,
#     "Chair"                 : 24,
#     "Bench"                 : 25,
#     "Table"                 : 26,
#     "Trash Can"             : 27,
#     "Newspaper Dispenser"   : 28,
#     # Static Classes Indoor
#     "Room Label"            : 29,
#     "Stanchion"             : 30,
#     "Sanitizer Dispenser"   : 31,
#     "Condiment Dispenser"   : 32,
#     "Vending Machine"       : 33,
#     "Emergency Aid Kit"     : 34,
#     "Fire Extinguisher"     : 35,
#     "Computer"              : 36,
#     "Television"            : 37, # unused
#     "Other"                 : 38,
#     "Horse"                 : 39,
#     # New Classes
#     "Pickup Truck"          : 40,
#     "Delivery Truck"        : 41,
#     "Service Vehicle"       : 42,
#     "Utility Vehicle"       : 43,
#     "Fire Alarm"            : 44,
#     "ATM"                   : 45,
#     "Cart"                  : 46,
#     "Couch"                 : 47,
#     "Traffic Arm"           : 48,
#     "Wall Sign"             : 49,
#     "Floor Sign"            : 50,
#     "Door Switch"           : 51,
#     "Emergency Phone"       : 52,
#     "Dumpster"              : 53,
#     "Vacuum Cleaner"        : 54, # unused
#     "Segway"                : 55,
#     "Bus"                   : 56,
#     "Skateboard"            : 57,
#     "Water Fountain"        : 58
# }

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


def create_coda_infos_split(root_path,
                            info_prefix,
                            out_dir,
                            split_meta,
                            split):
    
    metainfo = {
        'categories': BBOX_CLASS_TO_ID,
        'dataset': 'coda',
        'version': 'v1.0',
        'split': split}
    data_list = []
    count = 0
    total = split_meta[split]['frame_len']
    print(f'Processing {split} split with {total} samples')
    
    for scene in split_meta[split]['scenes']:
        scene_idx = scene['scene']
        frame_list = scene['frames']
        timestamp_list = open(os.path.join(root_path, 'timestamps', f'{scene_idx}.txt')).readlines()
        pose_list = open(os.path.join(root_path, 'poses', 'dense', f'{scene_idx}.txt')).readlines()
        calibs = {}
        for cam in ['cam0', 'cam1']:
            calibs[cam] = {
                'intrinsics': yaml.load(open(os.path.join(root_path, 'calibrations', 
                            scene_idx, f'calib_{cam}_intrinsics.yaml')), Loader=yaml.FullLoader),
                'os2cam': yaml.load(open(os.path.join(root_path, 'calibrations', 
                            scene_idx, f'calib_os1_to_{cam}.yaml')), Loader=yaml.FullLoader)}
            
        for frame in frame_list:
            data = {}
            data['scene'] = scene_idx
            data['frame'] = frame
            data['sample_idx'] = count
            data['token'] = f'{scene_idx}_{frame}'
            data['timestamp'] = timestamp_list[frame].strip()
            data['lidar_points'] = {
                'num_pts_feats': 4,
                'lidar_path': os.path.join('3d_comp', 'os1', scene_idx, f'3d_comp_os1_{scene_idx}_{frame}.bin')}
            
            images = {}
            for cam in ['cam0', 'cam1']:
                images[cam] = {}
                images[cam]['img_path'] = os.path.join('2d_rect', cam, scene_idx, f'2d_rect_{cam}_{scene_idx}_{frame}.png')
                images[cam]['cam2img'] = np.array(calibs[cam]['intrinsics']['projection_matrix']['data']).reshape(3, 4)[:, :3]
                rect = np.eye(4)
                rect[:3, :3] = np.array(calibs[cam]['intrinsics']['rectification_matrix']['data']).reshape(3, 3)
                T = np.array(calibs[cam]['os2cam']['extrinsic_matrix']['data']).reshape(4, 4)
                images[cam]['lidar2cam'] = rect @ T
                # images[cam]['lidar2img'] = images[cam]['cam2img'] @ images[cam]['lidar2cam']
                images[cam]['timestamp'] = data['timestamp']
            data['images'] = images
            
            instances = []
            bboxes = json.load(open(os.path.join(root_path, '3d_bbox', 'os1', scene_idx, f'3d_bbox_os1_{scene_idx}_{frame}.json')))
            for bbox in bboxes['3dbbox']:
                if bbox['classId'] not in BBOX_CLASS_REMAP:
                    continue
                instance = {}
                instance['bbox_label_3d'] = BBOX_CLASS_TO_ID[BBOX_CLASS_REMAP[bbox['classId']]]
                if instance['bbox_label_3d'] == 2 and bbox['h'] < 1.5:
                    continue
                instance['bbox_3d'] = [bbox['cX'], bbox['cY'], bbox['cZ'], bbox['l'], bbox['w'], bbox['h'], bbox['y']]
                instance['is_occluded'] = bbox['labelAttributes']['isOccluded']
                instance['instance_id'] = bbox['instanceId']
                instances.append(instance)
            data['instances'] = instances
            
            ego2global = np.array(pose_list[frame].split()[1:]) # [x, y, z, qw, qx, qy, qz]
            ego2global_r = Quaternion(ego2global[3:]).rotation_matrix
            data['ego2global'] = np.eye(4)
            data['ego2global'][:3, :3] = ego2global_r
            data['ego2global'][:3, 3] = ego2global[:3]
            
            data_list.append(data)
            count += 1
            if count % 100 == 0:
                print(f'{count}/{total} samples processed')
    
    info_path = os.path.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
    mmengine.dump({'metainfo': metainfo, 'data_list': data_list}, info_path)
            

def create_coda_infos(root_path,
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
    split_file = os.path.join(root_path, 'split.json')
    split_meta = json.load(open(split_file, 'r'))
    
    if version == 'v1.0-trainval':
        create_coda_infos_split(root_path, info_prefix, out_dir, split_meta, split='train')
        create_coda_infos_split(root_path, info_prefix, out_dir, split_meta, split='val')
    elif version == 'v1.0-train':
        create_coda_infos_split(root_path, info_prefix, out_dir, split_meta, split='train')
    elif version == 'v1.0-val':
        create_coda_infos_split(root_path, info_prefix, out_dir, split_meta, split='val')
    elif version == 'v1.0-test':
        create_coda_infos_split(root_path, info_prefix, out_dir, split_meta, split='test')