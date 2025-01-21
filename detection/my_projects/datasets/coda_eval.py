# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import gc
import io as sysio

import numba
import numpy as np


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    ignored_gt, ignored_dt = [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno['name'][i].lower()
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == 'Pedestrian'.lower()
              and 'Person_sitting'.lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == 'Car'.lower() and 'Van'.lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        # if ((gt_anno['occluded'][i] > MAX_OCCLUSION[difficulty])
        #         or (gt_anno['truncated'][i] > MAX_TRUNCATION[difficulty])
        #         or (height <= MIN_HEIGHT[difficulty])):
        #     ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt


def bev_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lidar.
    # TODO: change to use prange for parallel mode, should check the difference
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in numba.prange(N):
        for j in numba.prange(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (
                    min(boxes[i, 2], qboxes[j, 2]) -
                    max(boxes[i, 2] - boxes[i, 5],
                        qboxes[j, 2] - qboxes[j, 5]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    from .rotate_iou import rotate_iou_gpu_eval
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]],
                               qboxes[:, [0, 1, 3, 4, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_size,
                           dt_size,
                           dt_scores,
                           ignored_gt,
                           ignored_det,
                           min_overlap,
                           thresh=0,
                           compute_fp=False):
    assigned_detection = [False] * dt_size
    ignored_threshold = [False] * dt_size
    if compute_fp:
        for i in range(dt_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn = 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(dt_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(dt_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
    return tp, fp, fn, thresholds[:thresh_idx]


def get_split_parts(num, num_part):
    if num % num_part == 0:
        same_part = num // num_part
        return [same_part] * num_part
    else:
        same_part = num // (num_part - 1)
        remain_num = num % (num_part - 1)
        return [same_part] * (num_part - 1) + [remain_num]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             gt_sizes,
                             dt_sizes,
                             dt_scores_list,
                             ignored_gts,
                             ignored_dets,
                             min_overlap,
                             thresholds):
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i],
                               gt_num:gt_num + gt_nums[i]]
            gt_size = gt_sizes[i]
            dt_size = dt_sizes[i]
            dt_scores = dt_scores_list[i]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            tp, fp, fn, _ = compute_statistics_jit(
                overlap,
                gt_size,
                dt_size,
                dt_scores,
                ignored_gt,
                ignored_det,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]


def calculate_iou_partly(dt_annos, gt_annos, metric, num_parts=50):
    """Fast iou algorithm. this function can be used independently to do result
    analysis. Must be used in CAMERA coordinate system.

    Args:
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d.
        num_parts (int): A parameter for fast calculate algorithm.
    """
    assert len(dt_annos) == len(gt_annos)
    total_gt_num = np.stack([len(a['name']) for a in gt_annos], 0)
    total_dt_num = np.stack([len(a['name']) for a in dt_annos], 0)
    num_examples = len(dt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        if metric == 1:
            loc = np.concatenate([a['location'][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a['location'][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],axis=1)
            overlap_part = bev_box_overlap(dt_boxes, gt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a['location'] for a in dt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in dt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a['location'] for a in gt_annos_part], 0)
            dims = np.concatenate([a['dimensions'] for a in gt_annos_part], 0)
            rots = np.concatenate([a['rotation_y'] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(dt_boxes, gt_boxes).astype(np.float64)
        else:
            raise ValueError('unknown metric')
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][dt_num_idx:dt_num_idx + dt_box_num,
                                   gt_num_idx:gt_num_idx + gt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_dt_num, total_gt_num


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    gt_sizes = []
    dt_sizes = []
    dt_scores_list = []
    ignored_gts, ignored_dets = [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        total_num_valid_gt += num_valid_gt
        gt_sizes.append(len(gt_annos[i]['name']))
        dt_sizes.append(len(dt_annos[i]['name']))
        dt_scores_list.append(np.array(dt_annos[i]['score'], dtype=np.float64))
    return (gt_sizes, dt_sizes, dt_scores_list, ignored_gts, ignored_dets, total_num_valid_gt)


def eval_class(gt_annos,
               dt_annos,
               current_classes,
               difficultys,
               metric,
               min_overlaps,
               num_parts=200):
    """Kitti eval. support 2d/bev/3d/aos eval. support 0.5:0.05:0.95 coco AP.

    Args:
        gt_annos (dict): Must from get_label_annos() in kitti_common.py.
        dt_annos (dict): Must from get_label_annos() in kitti_common.py.
        current_classes (list[int]): 0: car, 1: pedestrian, 2: cyclist.
        difficultys (list[int]): Eval difficulty, 0: easy, 1: normal, 2: hard
        metric (int): Eval type. 0: bbox, 1: bev, 2: 3d
        min_overlaps (float): Min overlap. format:
            [num_overlap, metric, class].
        num_parts (int): A parameter for fast calculate algorithm

    Returns:
        dict[str, np.ndarray]: recall, precision and aos
    """
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros(
        [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(current_classes):
        for idx_l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            (gt_sizes, dt_sizes, dt_scores_list, ignored_gts, ignored_dets, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    rets = compute_statistics_jit(
                        overlaps[i],
                        gt_sizes[i],
                        dt_sizes[i],
                        dt_scores_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        min_overlap=min_overlap,
                        thresh=0.0,
                        compute_fp=False)
                    tp, fp, fn, thresholds = rets
                    thresholdss += thresholds.tolist()
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    ignored_dets_part = np.concatenate(
                        ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(
                        ignored_gts[idx:idx + num_part], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        gt_sizes[idx:idx + num_part],
                        dt_sizes[idx:idx + num_part],
                        dt_scores_list[idx:idx + num_part],
                        ignored_gts_part,
                        ignored_dets_part,
                        min_overlap=min_overlap,
                        thresholds=thresholds)
                    idx += num_part
                for i in range(len(thresholds)):
                    recall[m, idx_l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, idx_l, k, i] = pr[i, 0] / (
                        pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, idx_l, k, i] = np.max(
                        precision[m, idx_l, k, i:], axis=-1)
                    recall[m, idx_l, k, i] = np.max(
                        recall[m, idx_l, k, i:], axis=-1)
    ret_dict = {
        'recall': recall,
        'precision': precision,
    }

    # clean temp variables
    del overlaps
    del parted_overlaps

    gc.collect()
    return ret_dict


def get_mAP11(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            eval_types=['bev', '3d']):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [0, 1, 2]

    mAP11_bev = None
    mAP40_bev = None
    if 'bev' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                         min_overlaps)
        mAP11_bev = get_mAP11(ret['precision'])
        mAP40_bev = get_mAP40(ret['precision'])

    mAP11_3d = None
    mAP40_3d = None
    if '3d' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                         min_overlaps)
        mAP11_3d = get_mAP11(ret['precision'])
        mAP40_3d = get_mAP40(ret['precision'])
    return (mAP11_bev, mAP11_3d, mAP40_bev, mAP40_3d)


def kitti_eval(gt_annos,
               dt_annos,
               current_classes,
               eval_types=['bev', '3d']):
    """KITTI evaluation.

    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].

    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, 'must contain at least one evaluation type'
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5], [0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.7, 0.5, 0.5, 0.7, 0.5]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25],
                            [0.5, 0.25, 0.25, 0.5, 0.25]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''

    mAP11_bev, mAP11_3d, mAP40_bev, mAP40_3d = do_eval(gt_annos, dt_annos,
                                      current_classes, min_overlaps, eval_types)

    ret_dict = {}
    difficulty = ['easy', 'moderate', 'hard']

    # calculate AP11
    result += '\n----------- AP11 Results ------------\n\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP11@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j]))
            if mAP11_bev is not None:
                result += 'bev  AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP11_bev[j, :, i])
            if mAP11_3d is not None:
                result += '3d   AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP11_3d[j, :, i])

            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'
                prefix = f'KITTI/{curcls_name}'
                if mAP11_3d is not None:
                    ret_dict[f'{prefix}_3D_AP11_{postfix}'] =\
                        mAP11_3d[j, idx, i]
                if mAP11_bev is not None:
                    ret_dict[f'{prefix}_BEV_AP11_{postfix}'] =\
                        mAP11_bev[j, idx, i]

    # calculate mAP11 over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP11@{}, {}, {}:\n'.format(*difficulty))
        if mAP11_bev is not None:
            mAP11_bev = mAP11_bev.mean(axis=0)
            result += 'bev  AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP11_bev[:, 0])
        if mAP11_3d is not None:
            mAP11_3d = mAP11_3d.mean(axis=0)
            result += '3d   AP11:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP11_3d[:, 0])

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            if mAP11_3d is not None:
                ret_dict[f'KITTI/Overall_3D_AP11_{postfix}'] = mAP11_3d[idx, 0]
            if mAP11_bev is not None:
                ret_dict[f'KITTI/Overall_BEV_AP11_{postfix}'] =\
                    mAP11_bev[idx, 0]

    # Calculate AP40
    result += '\n----------- AP40 Results ------------\n\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP40@{:.2f}, {:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j]))
            if mAP40_bev is not None:
                result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_bev[j, :, i])
            if mAP40_3d is not None:
                result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                    *mAP40_3d[j, :, i])

            # prepare results for logger
            for idx in range(3):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'
                prefix = f'KITTI/{curcls_name}'
                if mAP40_3d is not None:
                    ret_dict[f'{prefix}_3D_AP40_{postfix}'] =\
                        mAP40_3d[j, idx, i]
                if mAP40_bev is not None:
                    ret_dict[f'{prefix}_BEV_AP40_{postfix}'] =\
                        mAP40_bev[j, idx, i]

    # calculate mAP40 over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP40@{}, {}, {}:\n'.format(*difficulty))
        if mAP40_bev is not None:
            mAP40_bev = mAP40_bev.mean(axis=0)
            result += 'bev  AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(
                *mAP40_bev[:, 0])
        if mAP40_3d is not None:
            mAP40_3d = mAP40_3d.mean(axis=0)
            result += '3d   AP40:{:.4f}, {:.4f}, {:.4f}\n'.format(*mAP40_3d[:, 0])

        # prepare results for logger
        for idx in range(3):
            postfix = f'{difficulty[idx]}'
            if mAP40_3d is not None:
                ret_dict[f'KITTI/Overall_3D_AP40_{postfix}'] = mAP40_3d[idx, 0]
            if mAP40_bev is not None:
                ret_dict[f'KITTI/Overall_BEV_AP40_{postfix}'] =\
                    mAP40_bev[idx, 0]

    return result, ret_dict