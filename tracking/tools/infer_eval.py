import os
import yaml
import time
from copy import deepcopy
import numpy as np
import torch
import mmengine
from mot_3d.data_protos import BBox
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from mot_3d.update_info_data import UpdateInfoData
import mot_3d.tracklet as tracklet
from mot_3d.association import associate_dets_to_tracks, associate_unmatched_trks


def load_frame(data, detection_result, i, track_labels, point_cloud_range):
    assert data['sample_idx'] == detection_result['sample_idx'][i]
    # points = np.fromfile('./data/CODA/' + data['lidar_path'], dtype=np.float32).reshape([-1, 4])
    # mask = (points[:, 0] > point_cloud_range[0]) & (points[:, 0] < point_cloud_range[3]) & \
    #     (points[:, 1] > point_cloud_range[1]) & (points[:, 1] < point_cloud_range[4]) & \
    #     (points[:, 2] > point_cloud_range[2]) & (points[:, 2] < point_cloud_range[5])
    # points = points[mask]
    ego2global = data['ego2global']
    pre_bboxes = detection_result['pre_bboxes'][i]
    pre_labels = detection_result['pre_labels'][i]
    track_mask = np.isin(pre_labels, np.array(track_labels))
    pre_bboxes = pre_bboxes[track_mask].tolist()
    pre_labels = pre_labels[track_mask].tolist()
    frame_data = FrameData(dets=pre_bboxes, ego=ego2global, pc=None, det_types=pre_labels, time_stamp=data['time_stamp'])
    frame_data.dets = [BBox.bbox2world(ego2global, det) for det in frame_data.dets]
    
    gt_bboxes = [BBox.bbox2world(ego2global, BBox.array2bbox(instance['bbox_3d'])) for instance in data['instances']]
    gt_labels = [instance['bbox_label_3d'] for instance in data['instances']]
    gt_ids = [str(instance['instance_id'].split(':')[-1]) for instance in data['instances']]
    
    return frame_data, gt_bboxes, gt_labels, gt_ids


data_info = mmengine.load('./data/CODA/coda_infos_val.pkl')
detection_results = mmengine.load('./data/CODA/detection_results.pkl')
config_path = 'configs/nus_configs/diou.yaml'
configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
point_cloud_range = [-21.0, -21.0, -2.0, 21.0, 21.0, 6.0]

scene = -1
frame = -1
all_results = []
sequence_results = None
track_labels = [1, 2]

print('Processing %d frames' % len(data_info['data_list']))
for i, data in enumerate(data_info['data_list']):
    if data['scene'] != scene['scene_id'] or data['frame'] - 1 != frame:
        if sequence_results is not None:
            all_results.append(sequence_results)
            print('Processing scene %d end at frame %d (%d / %d)' % (scene, frame, i, len(data_info['data_list'])))
        scene = data['scene']
        frame = data['frame']
        print('Processing scene %d start from frame %d' % (scene, frame))
        tracker = MOTModel(configs)
        sequence_results = {'token': data['token'], 'data_list': []}
    
    frame_data, gt_bboxes, gt_labels, gt_ids = load_frame(data, detection_results, i, track_labels, point_cloud_range)
    results = tracker.frame_mot(frame_data)
    result_pred_bboxes = [trk[0] for trk in results]
    result_pred_ids = [trk[1] for trk in results]
    result_pred_states = [trk[2] for trk in results]
    result_labels = [trk[3] for trk in results]

    frame_result = {}
    frame_result['track_ids'] = result_pred_ids
    frame_result['track_bboxes'] = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
    frame_result['track_states'] = result_pred_states
    frame_result['track_labels'] = result_labels
    frame_result['gt_bboxes'] = [BBox.bbox2array(bbox) for bbox in gt_bboxes]
    frame_result['gt_labels'] = gt_labels
    frame_result['gt_ids'] = gt_ids
    sequence_results['data_list'].append(frame_result)

mmengine.dump(all_results, './data/CODA/code_track_result.pkl')