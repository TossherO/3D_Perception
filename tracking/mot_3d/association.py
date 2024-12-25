import numpy as np
import torch
from . import utils
from .data_protos import BBox
from scipy.optimize import linear_sum_assignment


def associate_dets_to_tracks(dets, tracks, mode, asso, 
    dist_threshold=0.9, trk_innovation_matrix=None):
    """ associate the tracks with detections
    """
    if mode == 'bipartite':
        matched_indices, dist_matrix = \
            bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    elif mode == 'greedy':
        matched_indices, dist_matrix = \
            greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix)
    unmatched_dets = list()
    for d, det in enumerate(dets):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)

    unmatched_tracks = list()
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)
    
    matches = list()
    for m in matched_indices:
        if dist_matrix[m[0], m[1]] > dist_threshold:
            unmatched_dets.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(2))
    return matches, np.array(unmatched_dets), np.array(unmatched_tracks)


def bipartite_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    gpu_device = None
    if torch.cuda.is_available():
        gpu_device = torch.cuda.current_device()
    if asso == 'iou' or asso == 'giou' or asso == 'diou':
        if gpu_device is not None and asso != 'giou':
            dist_matrix = compute_iou_distance_parallel(dets, tracks, asso, gpu_device)
        else:
            dist_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'm_dis':
        dist_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        dist_matrix = compute_m_distance(dets, tracks, None)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)
    return matched_indices, dist_matrix


def greedy_matcher(dets, tracks, asso, dist_threshold, trk_innovation_matrix):
    """ it's ok to use iou in bipartite
        but greedy is only for m_distance
    """
    matched_indices = list()
    
    # compute the distance matrix
    if asso == 'm_dis':
        distance_matrix = compute_m_distance(dets, tracks, trk_innovation_matrix)
    elif asso == 'euler':
        distance_matrix = compute_m_distance(dets, tracks, None)
    elif asso == 'iou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    elif asso == 'giou':
        distance_matrix = compute_iou_distance(dets, tracks, asso)
    num_dets, num_trks = distance_matrix.shape

    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_dets
    tracking_id_matches_to_detection_id = [-1] * num_trks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])
    if len(matched_indices) == 0:
        matched_indices = np.empty((0, 2))
    else:
        matched_indices = np.asarray(matched_indices)
    return matched_indices, distance_matrix


def compute_m_distance(dets, tracks, trk_innovation_matrix):
    """ compute l2 or mahalanobis distance
        when the input trk_innovation_matrix is None, compute L2 distance (euler)
        else compute mahalanobis distance
        return dist_matrix: numpy array [len(dets), len(tracks)]
    """
    euler_dis = (trk_innovation_matrix is None) # is use euler distance
    if not euler_dis:
        trk_inv_inn_matrices = [np.linalg.inv(m) for m in trk_innovation_matrix]
    dist_matrix = np.empty((len(dets), len(tracks)))

    for i, det in enumerate(dets):
        for j, trk in enumerate(tracks):
            if euler_dis:
                dist_matrix[i, j] = utils.m_distance(det, trk)
            else:
                dist_matrix[i, j] = utils.m_distance(det, trk, trk_inv_inn_matrices[j])
    return dist_matrix


def compute_iou_distance(dets, tracks, asso='iou'):
    iou_matrix = np.zeros((len(dets), len(tracks)))
    for d, det in enumerate(dets):
        for t, trk in enumerate(tracks):
            if asso == 'iou':
                iou_matrix[d, t] = utils.iou3d(det, trk)[1]
            elif asso == 'giou':
                iou_matrix[d, t] = utils.giou3d(det, trk)
    dist_matrix = 1 - iou_matrix
    return dist_matrix


def compute_iou_distance_parallel(dets, tracks, asso, gpu_device):
    dets = torch.tensor(np.array([BBox.bbox2array(det, return_score=False) for det in dets]), dtype=torch.float32, device=gpu_device)
    tracks = torch.tensor(np.array([BBox.bbox2array(trk, return_score=False) for trk in tracks]), dtype=torch.float32, device=gpu_device)
    if len(dets) == 0 or len(tracks) == 0:
        return np.zeros((len(dets), len(tracks)))
    dets = dets.unsqueeze(1).repeat(1, len(tracks), 1)
    tracks = tracks.unsqueeze(0).repeat(len(dets), 1, 1)
    if asso == 'iou':
        iou_matrix = utils.diff_iou_rotated_3d(dets, tracks)
    elif asso == 'diou':
        iou_matrix = utils.diff_diou_rotated_3d(dets, tracks)
    dist_matrix = 1 - iou_matrix
    return dist_matrix.cpu().numpy()


def associate_unmatched_trks(dets, pred_bboxes, asso, dist_threshold):
    if torch.cuda.is_available():
        gpu_device = torch.cuda.current_device()
    else:
        return None
    
    if asso == 'giou':
        update_modes = []
        for trk in pred_bboxes:
            max_iou = -1.0
            for det in dets:
                iou = utils.giou3d(det, trk)
                if iou > max_iou:
                    max_iou = iou
            if max_iou > dist_threshold:
                update_modes.append(3)
            else:
                update_modes.append(0)
        return update_modes

    dets = torch.tensor(np.array([BBox.bbox2array(det, return_score=False) for det in dets]), dtype=torch.float32, device=gpu_device)
    tracks = torch.tensor(np.array([BBox.bbox2array(trk, return_score=False) for trk in pred_bboxes]), dtype=torch.float32, device=gpu_device)
    if len(dets) == 0 or len(tracks) == 0:
        return [0] * len(pred_bboxes)
    dets = dets.unsqueeze(1).repeat(1, len(tracks), 1)
    tracks = tracks.unsqueeze(0).repeat(len(dets), 1, 1)
    if asso == 'iou':
        dists = utils.diff_iou_rotated_3d(dets, tracks)
    elif asso == 'diou':
        dists = utils.diff_diou_rotated_3d(dets, tracks)

    if asso in ['iou', 'giou', 'diou']:
        is_associated = torch.max(dists, dim=0)[0] > dist_threshold
    elif asso in ['m_dis', 'euler']:
        is_associated = torch.min(dists, dim=0)[0] < dist_threshold

    update_modes = []
    for i in range(len(is_associated)):
        if is_associated[i]:
            update_modes.append(3)
        else:
            update_modes.append(0)

    return update_modes