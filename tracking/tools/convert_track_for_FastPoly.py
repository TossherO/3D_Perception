import os
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))
import numpy as np
import mmengine
import json
from pyquaternion import Quaternion

data_info = mmengine.load('./data/CODA/coda_infos_val.pkl')
detection_results = mmengine.load('./data/CODA/detection_results.pkl')

scene = -1
frame = -1
first_tokens = []
converted_results = {}
class_names = ['car', 'pedestrian', 'cyclist']

for i, data in enumerate(data_info['data_list']):
    
    det_sample_idx = detection_results['sample_idx'][i]
    det_bboxes = detection_results['pre_bboxes'][i]
    det_labels = detection_results['pre_labels'][i]
    det_scores = detection_results['pre_scores'][i]
    assert data['sample_idx'] == det_sample_idx
    if data['scene'] != scene or data['frame'] - 1 != frame:
        first_tokens.append(data['token'])
    
    scene = data['scene']
    frame = data['frame']
    ego2global = data['ego2global']
    converted_result = []
    for j in range(len(det_bboxes)):
        bbox2ego = np.eye(4)
        bbox2ego[:3, :3] = Quaternion(axis=[0, 0, 1], angle=det_bboxes[j, 6]).rotation_matrix
        bbox2ego[:3, 3] = det_bboxes[j, :3]
        bbox2global = np.dot(ego2global, bbox2ego)
        translation = bbox2global[:3, 3].tolist()
        rotation = Quaternion(matrix=bbox2global[:3, :3]).elements.tolist()
        converted_result.append({
            'sample_token': data['token'],
            'translation': translation,
            'size': det_bboxes[j, 3:6].tolist(),
            'rotation': rotation,
            'velocity': [0, 0],
            'detection_name': class_names[det_labels[j]],
            'detection_score': float(det_scores[j]),
            'attribute_name': class_names[det_labels[j]]
        })

    converted_results[data['token']] = converted_result
    if i % 100 == 0:
        print(f'{i}/{len(data_info["data_list"])}')

meta = {
    'use_camera': True,
    'use_lidar': True,
    'use_radar': False,
    'use_map': False,
    'use_external': False
}

# 保存为json文件
output_path = './data/CODA/coda_val_detection.json'
with open(output_path, 'w') as f:
    json.dump({'meta': meta, 'results': converted_results}, f, indent=2)
first_tokens_path = './data/CODA/coda_val_first_token.json'
with open(first_tokens_path, 'w') as f:
    json.dump(first_tokens, f, indent=2)