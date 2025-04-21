import os
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))
import yaml
import time
import numpy as np
import torch
import mmengine
from mot_3d.data_protos import BBox
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from mmcv.ops.diff_iou_rotated import box2corners, oriented_box_intersection_2d


def load_frame(data, detection_result, i, label, point_cloud_range):
    assert data['sample_idx'] == detection_result['sample_idx'][i]
    # points = np.fromfile('./data/CODA/' + data['lidar_path'], dtype=np.float32).reshape([-1, 4])
    # mask = (points[:, 0] > point_cloud_range[0]) & (points[:, 0] < point_cloud_range[3]) & \
    #     (points[:, 1] > point_cloud_range[1]) & (points[:, 1] < point_cloud_range[4]) & \
    #     (points[:, 2] > point_cloud_range[2]) & (points[:, 2] < point_cloud_range[5])
    # points = points[mask]
    ego2global = data['ego2global']
    pre_labels = detection_result['pre_labels'][i]
    mask = pre_labels == label
    pre_bboxes = detection_result['pre_bboxes'][i][mask]
    pre_scores = detection_result['pre_scores'][i][mask]
    pre_labels = pre_labels[mask]
    dets = np.concatenate([pre_bboxes, pre_scores[:, None]], axis=1).tolist()
    frame_data = FrameData(dets=dets, ego=ego2global, pc=None, det_types=pre_labels, time_stamp=float(data['timestamp']))
    frame_data.dets = [BBox.bbox2world(ego2global, det) for det in frame_data.dets]
    
    gt_bboxes = []
    gt_labels = []
    gt_ids = []
    for instance in data['instances']:
        if instance['bbox_label_3d'] == label:
            gt_bboxes.append(BBox.bbox2world(ego2global, BBox.array2bbox(instance['bbox_3d'])))
            gt_labels.append(instance['bbox_label_3d'])
            gt_ids.append(instance['instance_id'])
    
    return frame_data, gt_bboxes, gt_labels, gt_ids


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


data_info = mmengine.load('./data/CODA/coda_infos_val.pkl')
detection_results = mmengine.load('./data/CODA/coda_cmdt_detection_results.pkl')
config_path = './configs/coda_configs/giou.yaml'
save_path = './work_dirs/' + time.strftime('%Y%m%d%H%M%S', time.localtime()) + '/'
configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
point_cloud_range = [-21.0, -21.0, -2.0, 21.0, 21.0, 6.0]

# infer
scene = -1
frame = -1
labels = [0, 1, 2]
class_names = ['car', 'pedestrian', 'cyclist']
trackers = [None for _ in labels]
sequence_results = None
start_time = time.time()

print('Processing %d frames' % len(data_info['data_list']))
for i, data in enumerate(data_info['data_list']):
    if data['scene'] != scene or data['frame'] - 1 != frame:
        if sequence_results is not None:
            print('Processing scene %s end at frame %s (%d / %d)' % (scene, frame, i, len(data_info['data_list'])))
        scene = data['scene']
        frame = data['frame']
        print('Processing scene %s start from frame %s' % (scene, frame))
        trackers = [MOTModel(configs, class_names[label]) for label in labels]
        sequence_results = {'token': data['token'], 'data_list': [[] for _ in labels]}
    
    scene = data['scene']
    frame = data['frame']
    for j, label in enumerate(labels):
        frame_data, gt_bboxes, gt_labels, gt_ids = load_frame(data, detection_results, i, label, point_cloud_range)
        results = trackers[j].frame_mot(frame_data)
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
        sequence_results['data_list'][j].append(frame_result)

print('Processing scene %s end at frame %s (%d / %d)' % (scene, frame, i+1, len(data_info['data_list'])))

end_time = time.time()
print('Processing time: %.4fs, Mean time: %.4fs' % (end_time - start_time, (end_time - start_time) / len(data_info['data_list'])))