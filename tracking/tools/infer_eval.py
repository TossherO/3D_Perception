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


data_info = mmengine.load('./data/CODA/coda_infos_val_full.pkl')
detection_results = mmengine.load('./data/CODA/coda_cmdt_detection_results.pkl')
config_path = './configs/coda_configs/diou.yaml'
save_path = './work_dirs/' + time.strftime('%Y%m%d%H%M%S', time.localtime()) + '/'
configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
point_cloud_range = [-21.0, -21.0, -2.0, 21.0, 21.0, 6.0]

# infer
scene = -1
frame = -1
all_results = []
labels = [0, 1, 2]
class_names = ['car', 'pedestrian', 'cyclist']
trackers = [None for _ in labels]
sequence_results = None

print('Processing %d frames' % len(data_info['data_list']))
for i, data in enumerate(data_info['data_list']):
    if data['scene'] != scene or data['frame'] - 1 != frame:
        if sequence_results is not None:
            all_results.append(sequence_results)
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

all_results.append(sequence_results)
print('Processing scene %s end at frame %s (%d / %d)' % (scene, frame, i+1, len(data_info['data_list'])))
mmengine.dump(all_results, save_path + 'coda_track_results.pkl')

# eval
tracks_list = [[] for _ in labels]
tracks_num_list = [[] for _ in labels]
tp_list = [[] for _ in labels]
fp_list = [[] for _ in labels]
fn_list = [[] for _ in labels]
id_switch_list = [[] for _ in labels]
mt_list = [[] for _ in labels]
ml_list = [[] for _ in labels]
iou_shreshold = [0.7, 0.5, 0.5]

for results in all_results:
    print('Processing %s' % results['token'])
    for i, label in enumerate(labels):
        class_results = results['data_list'][i]
        tracks = {}
        tp = 0
        fp = 0
        fn = 0
        mt = 0
        ml = 0
        id_switch = 0
        for frame in class_results:
            # 加载数据
            gt_bboxes = torch.tensor(np.array(frame['gt_bboxes']), dtype=torch.float).cuda()
            gt_ids = frame['gt_ids']
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

            # 匹配，并统计TP、FP、FN
            if len(gt_bboxes) > 0 and len(track_bboxes) > 0:
                # dist_matrix = torch.cdist(track_bboxes[:, :3], gt_bboxes[:, :3])
                iou_matrix = iou_rotated_2d(track_bboxes, gt_bboxes)
                matched_ids = np.zeros(len(gt_bboxes), dtype=int) - 1
                for k in range(len(track_bboxes)):
                    max_iou, max_idx = torch.max(iou_matrix[k], 0)
                    if max_iou > iou_shreshold[label]:
                        matched_ids[max_idx] = track_ids[k]
                        iou_matrix[:, max_idx] = 0
                        tp += 1
                    else:
                        fp += 1
                fn += len(gt_bboxes) - sum(matched_ids != -1)

                # 构建轨迹匹配序列
                for k in range(len(gt_bboxes)):
                    if tracks.get(gt_ids[k]) is None:
                        tracks[gt_ids[k]] = []
                    tracks[gt_ids[k]].append(matched_ids[k])
                    
            else:
                fp += len(track_bboxes)
                fn += len(gt_bboxes)

        # 计算ID切换、MT、ML
        for track in tracks.values():
            tracked_num = 0
            last_id = -1
            for id in track:
                if id != -1:
                    if last_id != -1 and last_id != id:
                        id_switch += 1
                    last_id = id
                    tracked_num += 1
            if tracked_num / len(track) > 0.8:
                mt += 1
            elif tracked_num / len(track) < 0.2:
                ml += 1

        tracks_list[i].append(tracks)
        tracks_num_list[i].append(len(tracks))
        tp_list[i].append(tp)
        fp_list[i].append(fp)
        fn_list[i].append(fn)
        id_switch_list[i].append(id_switch)
        mt_list[i].append(mt)
        ml_list[i].append(ml)

# 计算MOTA、MT、ML
for i, label in enumerate(labels):
    mota = 1 - (sum(fp_list[i]) + sum(fn_list[i]) + sum(id_switch_list[i])) / (sum(tp_list[i]) + sum(fn_list[i]))
    mt = sum(mt_list[i]) / sum(tracks_num_list[i])
    ml = sum(ml_list[i]) / sum(tracks_num_list[i])
    print('Label %d: MOTA: %.4f, MT: %.4f, ML: %.4f' % (label, mota, mt, ml))
    print('TP: %d, FP: %d, FN: %d, ID Switch: %d' % (sum(tp_list[i]), sum(fp_list[i]), sum(fn_list[i]), sum(id_switch_list[i])))
    with open(save_path + 'coda_track_eval_results.log', 'a') as f:
        f.write('Label %d: MOTA: %.4f, MT: %.4f, ML: %.4f\n' % (label, mota, mt, ml))
        f.write('TP: %d, FP: %d, FN: %d, ID Switch: %d\n' % (sum(tp_list[i]), sum(fp_list[i]), sum(fn_list[i]), sum(id_switch_list[i])))