import os
import sys
import os.path as osp
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./detection'))
sys.path.append(os.path.abspath('./tracking'))
import time
import yaml
import json
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import mmengine
import open3d as o3d
from pyquaternion import Quaternion
from mmengine.config import Config
from mmdet3d.registry import MODELS, TRANSFORMS
from mmdet3d.utils import register_all_modules
from mmengine.runner import load_checkpoint
from mmdet3d.structures import LiDARPoints
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures import Box3DMode
from tracking.mot_3d.mot import MOTModel
from tracking.mot_3d.data_protos import BBox
from tracking.mot_3d.frame_data import FrameData
from trajectory_prediction.tools.dataset import TrajectoryDataset
from trajectory_prediction.model.model import TrajectoryModel


def data_preprocess(tracks, config):
    
    obs_len = config['obs_len']
    nei_radius = config['nei_radius']
    obs = []
    neis_ = []
    num_neis = []
    self_labels = []
    nei_labels_ = []
    refs = []
    rot_mats = []
    bboxes = []
    all_label_ids = list(tracks.keys())
    all_labels = np.array([int(label_id.split('_')[0]) for label_id in all_label_ids])
    all_tracks = np.array([tracks[k]['data'] for k in all_label_ids])
    all_bboxes = np.array([tracks[k]['bbox'] for k in all_label_ids])

    for i in range(len(all_tracks)):
        if all_tracks[i][-1][0] > 1e8 or all_tracks[i][obs_len-2][0] > 1e8 or all_labels[i] == 2:
            continue
        ob = all_tracks[i].copy()
        for j in range(obs_len - 2, -1, -1):
            if ob[j][0] > 1e8:
                ob[j] = ob[j+1]
        nei = all_tracks[np.arange(len(all_tracks)) != i]
        nei_labels = all_labels[np.arange(len(all_labels)) != i]
        now_nei_radius = [nei_radius[label] for label in nei_labels]
        dist_threshold = np.maximum(nei_radius[all_labels[i]], now_nei_radius)
        dist = np.linalg.norm(ob[:obs_len].reshape(1, obs_len, 2) - nei, axis=-1)
        dist = np.min(dist, axis=-1)
        nei = nei[dist < dist_threshold]
        nei_labels = nei_labels[dist < dist_threshold]
        
        ref = ob[-1]
        ob = ob - ref
        if nei.shape[0] != 0:
            nei = nei - ref
        angle = np.arctan2(ob[0][1], ob[0][0])
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], 
                            [np.sin(angle), np.cos(angle)]])
        ob = np.dot(ob, rot_mat)
        if nei.shape[0] != 0:
            nei = np.dot(nei, rot_mat)
        
        obs.append(ob)
        neis_.append(nei)
        num_neis.append(nei.shape[0])
        self_labels.append(all_labels[i])
        nei_labels_.append(nei_labels)
        refs.append(ref.flatten())
        rot_mats.append(rot_mat)
        bboxes.append(all_bboxes[i])
        
    if len(obs) == 0:
        return None
            
    max_num_nei = max(num_neis)
    if max_num_nei == 0:
        max_num_nei = 1
    nei_masks = torch.zeros(len(obs), max_num_nei, dtype=torch.bool)
    neis = torch.zeros(len(obs), max_num_nei, obs_len, 2)
    nei_labels = torch.zeros(len(obs), max_num_nei, dtype=torch.int32) - 1
    for i in range(len(obs)):
        nei_masks[i, :num_neis[i]] = True
        neis[i, :num_neis[i]] = torch.tensor(neis_[i])
        nei_labels[i, :num_neis[i]] = torch.tensor(nei_labels_[i])
    
    obs = torch.tensor(np.stack(obs, axis=0), dtype=torch.float32)
    self_labels = torch.tensor(self_labels, dtype=torch.int32) + 1
    refs = torch.tensor(np.stack(refs, axis=0), dtype=torch.float32)
    rot_mats = torch.tensor(np.stack(rot_mats, axis=0), dtype=torch.float32)
    bboxes = torch.tensor(np.stack(bboxes, axis=0), dtype=torch.float32)
    return obs, neis, nei_masks, self_labels, nei_labels, refs, rot_mats, bboxes


def update_tracks(tracks, labels, ids, xys, bboxes, config):

    obs_len = config['obs_len']
    is_updated = {k: False for k in tracks.keys()}
    for i in range(len(labels)):
        label_id = str(labels[i]) + '_' + str(ids[i])
        if tracks.get(label_id) is None:
            tracks[label_id] = {
                'data': [[1e9, 1e9] for _ in range(obs_len - 1)] + [xys[i].tolist()],
                'bbox': bboxes[i],
                'label': labels[i], 
                'lost_frame': 0}
        else:
            tracks[label_id]['data'].pop(0)
            tracks[label_id]['data'].append(xys[i].tolist())
            tracks[label_id]['bbox'] = bboxes[i]
            tracks[label_id]['lost_frame'] = 0
            is_updated[label_id] = True

    for k in is_updated.keys():
        if not is_updated[k]:
            if tracks[k]['lost_frame'] < obs_len:
                tracks[k]['data'].pop(0)
                tracks[k]['data'].append([1e9, 1e9])
                tracks[k]['lost_frame'] += 1
            else:
                del tracks[k]
    return tracks


def smooth_trajectories(trajectories, window_size=3):

    # trajectories: tensor(bs, num, length, 2)
    smoothed_trajectories = trajectories.clone()
    for i in range(trajectories.shape[-2]):
        if i < window_size:
            smoothed_trajectories[..., i, :] = torch.mean(trajectories[..., :i+window_size+1, :], dim=-2)
        elif i > trajectories.shape[2] - window_size - 1:
            smoothed_trajectories[..., i, :] = torch.mean(trajectories[..., i-window_size:, :], dim=-2)
        else:
            smoothed_trajectories[..., i, :] = torch.mean(trajectories[..., i-window_size:i+window_size+1, :], dim=2)
    return smoothed_trajectories


# load detection model
cfg = Config.fromfile('./detection/my_projects/CMDT/configs/cmdt_wheelchair.py')
checkpoint = './detection/ckpts/CMDT/cmdt_wheelchair.pth'
register_all_modules()
detect_model = MODELS.build(cfg.model)
pipeline = []
for transform in cfg.test_dataloader.dataset.pipeline:
    pipeline.append(TRANSFORMS.build(transform))
checkpoint = load_checkpoint(detect_model, checkpoint, map_location='cpu')
detect_model.cuda().eval()

# load tracking model
config_path = './tracking/configs/coda_configs/diou.yaml'
track_config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
class_labels = [0, 1]
class_names = ['pedestrian', 'cyclist']
trackers = [MOTModel(track_config, class_names[label]) for label in class_labels]

# load trajectory prediction model
config_path = './trajectory_prediction/configs/coda.yaml'
checkpoint = './trajectory_prediction/checkpoints/coda_best.pth'
traj_pred_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
traj_pred_model = TrajectoryModel(num_class=traj_pred_config['num_class'], in_size=2, 
                obs_len=traj_pred_config['obs_len'], pred_len=traj_pred_config['pred_len'], 
                embed_size=traj_pred_config['embed_size'], num_decode_layers=traj_pred_config['num_decode_layers'], 
                scale=traj_pred_config['scale'], pred_single=False)
traj_pred_model.load_state_dict(torch.load(checkpoint))
traj_pred_model.cuda().eval()

# load data_info
# data_info = mmengine.load(info_path)
data_path = './data/indoor2/'
tracks = {}
count = 1
frame_num = len(os.listdir(data_path + 'imgs'))

# initialize visualization
vis = o3d.visualization.Visualizer()
vis.create_window()
render_option = vis.get_render_option()
render_option.background_color = np.array([0, 0, 0])
render_option.point_size = 1
render_option.line_width = 2
ctr = vis.get_view_control()
label_colors = [[0, 1, 0], [0, 0, 1]]

# [250, 550], [1000, 1250], [1725, 2025]
process = {
    271: [[12.812600135803223, -5.446788311004639, 1]],
    277: [[4.701160430908203, 2.5170392990112305, -1]],
    279: [[10.857378959655762, -6.416009902954102, 0]],
    283: [[10.470844268798828, -6.052268028259277, 0]],
    289: [[5.063281059265137, 2.644129753112793, 1]],
    290: [[5.05638313293457, 2.8694000244140625, 1]],
    291: [[13.337339401245117, -4.986374378204346, 1]],
    297: [[2.7147066593170166, 2.6331405639648438, -1]],
    301: [[2.401336669921875, 2.7104625701904297, -1]],
    303: [[11.397555351257324, -1.5829849243164062, 1]],
    311: [[7.47540283203125, -5.2206926345825195, 1]],
    333: [[13.674161911010742, 4.533500671386719, -1]],
    334: [[13.657671928405762, 4.457642555236816, -1]],
    341: [[12.783791542053223, -1.4935302734375, 1]],
    342: [[12.274246215820312, -1.521010398864746, 1]],
    343: [[11.729525566101074, -1.54583740234375, 1]],
    344: [[11.267334938049316, -1.6526174545288086, 1]],
    345: [[10.820184707641602, -1.672478199005127, 1]],
    346: [[10.266047477722168, -1.6506500244140625, 1]],
    347: [[9.820871353149414, -1.647690773010254, 1]],
    348: [[9.337677001953125, -1.5885705947875977, 1]],
    350: [[7.55959939956665, 2.5791711807250977, 0]],
    354: [[7.511166095733643, 2.947620391845703, 0], [11.923874855041504, -5.509308815002441, 1]],
    362: [[3.488041639328003, -5.148205280303955, 1]],
    364: [[1.026564359664917, 5.2432708740234375, 1]],
    380: [[1.1351686716079712, -6.9896321296691895, 1]],
    464: [[4.783117771148682, -7.106035232543945, 1]],
    508: [[4.192745685577393, -5.325722694396973, 1]],
    532: [[5.819450855255127, -4.262083053588867, 1]],
    537: [[8.55690860748291, -4.363855361938477, 1]],
    995: [[3.7133989334106445, 3.812138557434082, -1]],
    996: [[3.4977169036865234,  4.02616024017334, -1]],
    997: [[3.4312500953674316, 3.993197441101074, -1]],
    998: [[3.34785532951355, 3.973094940185547, -1]],
    999: [[3.2425265312194824, 3.974308967590332, -1]],
    1000: [[3.1956770420074463, 3.9450483322143555, -1], [2.7255289554595947, -2.872842311859131, 0]],
    1001: [[3.045931100845337, 3.9395132064819336, -1], [6.758004188537598, 0.8907070159912109, 0]],
    1002: [[2.9662725925445557, 3.9380130767822266, -1]],
    1003: [[2.8323371410369873, 3.882373809814453, -1]],
    1004: [[2.6870155334472656, 3.8852615356445312, -1]],
    1005: [[2.715644598007202, 3.9401378631591797, -1]],
    1006: [[2.504082441329956, 3.9427976608276367, -1]],
    1007: [[2.3551151752471924, 3.9160995483398438, -1]],
    1008: [[2.2404391765594482, 3.8916215896606445, -1]],
    1009: [[2.1168837547302246, 3.8771543502807617, -1]],
    1010: [[1.9183117151260376, 3.8856277465820312, -1]],
    1011: [[1.7995009422302246, 3.79105281829834, -1]],
    1012: [[1.6956473588943481, 3.788577079772949, -1]],
    1013: [[1.6213054656982422, 3.7753944396972656, -1]],
    1014: [[1.3840352296829224, 3.790311813354492, -1]],
    1015: [[1.4891821146011353, 3.842137336730957, -1]],
    1016: [[1.3456666469573975, 3.8666610717773438, -1]],
    1017: [[1.2328062057495117, 3.8476791381835938, -1]],
    1018: [[1.1080459356307983, 3.8208370208740234, -1]],
    1019: [[0.9949734807014465, 3.8065223693847656, -1]],
    1020: [[0.9034761190414429, 3.7913875579833984, -1]],
    1021: [[0.8655083179473877, 3.778238296508789, -1]],
    1022: [[0.7384582757949829, 3.785244941711426, -1]],
    1023: [[0.6558067798614502, 3.7880258560180664, -1]],
    1024: [[0.6039333939552307, 3.729726791381836, -1]],
    1025: [[0.5674658417701721, 3.6753034591674805, -1]],
    1026: [[0.3233109414577484, 3.713991165161133, -1]],
    1027: [[0.16390520334243774, 3.620112419128418, -1]],
    1059: [[8.414799690246582, -5.153021335601807, 1]],
    1065: [[5.688922882080078, 5.077679634094238, -1]],
    1105: [[13.485633850097656, 3.37624454498291, -1]],
    1113: [[11.610962867736816, -3.9028854370117188, 1]],
    1132: [[14.152921676635742, -4.603385925292969, -1]],
    1133: [[9.093619346618652, 1.2868576049804688, -1]],
    1134: [[9.12533187866211, 1.5217742919921875, -1]],
    1135: [[1.4524503946304321, 2.063119888305664, -1]],
    1147: [[6.068753719329834, 0.3179931640625, 0]],
    1166: [[0.9565579891204834, -1.6340336799621582, 0]],
    1181: [[2.65925669670105, -4.3817667961120605, 1]],
    1185: [[0.3464144170284271, 4.312949180603027, 0]],
    1186: [[0.152324378490448, 4.327727317810059, 0]],
    1193: [[2.7259788513183594, 3.888479232788086, 1]],
    1194: [[2.60981822013855, 4.151383399963379, 1]],
    1195: [[2.4986183643341064, 4.405572891235352, 1]],
    1196: [[2.4023802280426025, 4.63817024230957, 1]],
    1198: [[8.572137832641602, -4.802537441253662, 0]],
    1200: [[10.485148429870605, -0.5283727645874023, -1]],
    1209: [[9.858805656433105, -3.544189453125, 0]],
    1223: [[7.44777250289917, 2.9102725982666016, -1]],
    1230: [[11.662321090698242, -1.9894185066223145, 0]],
    1235: [[12.185013771057129, -1.2992401123046875, 0]],
    1236: [[12.236272811889648, -1.0644197463989258, 0]],
    1237: [[12.299692153930664, -0.6887092590332031, 0]],
    1238: [[12.492823600769043, -0.18785476684570312, 0]],
    1239: [[12.530251502990723, 0.17276382446289062, 0]],
    1240: [[12.440446853637695, 0.5593633651733398, 0]],
    1242: [[12.578218460083008, 1.3022642135620117, 0]],
    1243: [[12.591838836669922, 1.3586091995239258, 0]],
    1247: [[8.970593452453613, -1.8682270050048828, 1]],
    1725: [[7.597284317016602, 4.428483963012695, -1]],
    1726: [[7.555185317993164, 4.429811477661133, -1]],
    1727: [[5.301085472106934, 3.944622039794922, -1]],
    1757: [[4.732481956481934, 3.240790367126465, -1]],
    1770: [[13.61434555053711, 0.6598148345947266, 0]],
    1771: [[13.348051071166992, 0.7044515609741211, 0]],
    1772: [[13.188528060913086, 0.6872739791870117, 0]],
    1773: [[12.972575187683105, 0.7130947113037109, 0]],
    1774: [[12.884145736694336, 0.6855058670043945, 0]],
    1775: [[12.617768287658691, 0.6345357894897461, 0]],
    1776: [[12.412708282470703, 0.5886077880859375, 0]],
    1777: [[12.16441535949707, 0.6006031036376953, 0]],
    1778: [[11.963375091552734, 0.628082275390625, 0]],
    1779: [[11.624473571777344, 0.6859054565429688, 0]],
    1782: [[11.109395980834961, 0.7287769317626953, 0]],
    1783: [[10.963017463684082, 0.7982578277587891, 0]],
    1784: [[10.722376823425293, 0.7370548248291016, 0]],
    1787: [[9.9642915725708, 0.685002326965332, 0]],
    1788: [[9.811004638671875, 0.7858142852783203, 0]],
    1789: [[9.548873901367188, 0.8446063995361328, 0]],
    1793: [[8.639281272888184, 0.9940071105957031, 0]],
    1795: [[9.718829154968262, -4.776611328125, -1]],
    1799: [[9.374412536621094, -4.654201507568359, -1]],
    1813: [[11.791545867919922, 1.3375177383422852, 0]],
    1820: [[9.91811752319336, 1.1659927368164062, 0]],
    1821: [[9.748710632324219, 1.1567230224609375, 0]],
    1860: [[7.570802211761475, 4.736950874328613, -1]],
    1872: [[1.9378821849822998, 1.140146255493164, 0]],
    1903: [[4.017518997192383, 4.423325538635254, -1]],
    1915: [[7.04109001159668, -1.976806640625, 1]],
    1916: [[2.9399495124816895, 4.34101676940918, -1]],
    1917: [[7.522662162780762, -2.0194268226623535, -1]],
    1920: [[8.027894973754883, -1.9884562492370605, 1]],
    1921: [[8.197774887084961, -2.002516269683838, 1]],
    1922: [[8.375457763671875, -2.011598587036133, 1]],
    1924: [[8.628031730651855, -1.990293025970459, 1], [14.12035083770752, 0.6248111724853516, 0]],
    1925: [[13.838064193725586, 0.6554851531982422, 0]],
    1926: [[13.64122486114502, 0.5793638229370117, 0]],
    1934: [[11.615571022033691, 0.3399333953857422, 0], [1.1521111726760864, 4.2928314208984375, -1]],
    1935: [[11.334415435791016, 0.3296833038330078, 0], [1.017162561416626, 4.268673896789551, -1]],
    1937: [[10.746903419494629, 0.1990642547607422, 0]],
    1938: [[10.479079246520996, 0.1333484649658203, 0]],
    1939: [[10.226924896240234, 0.09234333038330078, 0]],
    1940: [[9.992074966430664, 0.09843254089355469, 0]],
    1941: [[9.730904579162598, 0.037949562072753906, 0], [8.096953392028809, -1.817854404449463, 1]],
    1943: [[7.8407721519470215, -1.8243017196655273, 1]],
    1944: [[8.95633316040039, -0.09961128234863281, 0]],
    1945: [[8.634202003479004, -0.21998310089111328, 0]],
    1946: [[7.5550408363342285, -1.8835334777832031, 1]],
    1947: [[7.417415618896484, -1.900033950805664, 1]],
    1948: [[7.350927829742432, -1.9218106269836426, 1]],
    1949: [[7.259080410003662, -1.9208455085754395, 1]],
    1950: [[7.190299987792969, -1.9030141830444336, 1]],
    1951: [[7.001834392547607, -1.8753318786621094, 1]],
    1952: [[6.951608180999756, -1.8720273971557617, 1]],
    1953: [[7.013031005859375, -1.8480463027954102, 1]],
    1955: [[6.869143486022949, -1.8194756507873535, 1]],
    1957: [[6.671769142150879, -1.86509370803833, 1]],
    1967: [[4.678530216217041, -3.2103476524353027, 1]],
    1976: [[6.281863212585449, -3.3519177436828613, 1]],
    1979: [[10.589491844177246, -0.06316471099853516, 1], [3.8525278568267822, -0.9730329513549805, 0]],
    1983: [[9.68183708190918, -0.1665477752685547, 0]],
    1984: [[9.386634826660156, -0.11944198608398438, 0]],
    1985: [[9.162129402160645, -0.15674591064453125, 0]],
    1988: [[8.39127254486084, -0.18535900115966797, 0]],
    1990: [[5.170616149902344, -8.600285530090332, -1], [8.190587997436523, -4.106993675231934, 0]],
    2000: [[5.958107948303223, -0.6607608795166016, 0]]
}

# main loop
infer_time = 0
while count <= frame_num:
    
    start_time = time.time()
    #  load data
    input = {}
    points = np.fromfile(data_path + 'points/%04d.bin' % count, dtype=np.float32).reshape(4, -1).T
    imgs = [cv2.imread(data_path + 'imgs/%04d.png' % count)]
    # with open(data_path + 'pose/%04d.txt' % count, 'r') as f:
    #     pose = list(map(float, f.readlines()[0].strip().split()[1:]))
    input['points'] = LiDARPoints(points, points_dim=4)
    input['img'] = imgs
    input['timestamp'] = count / 10
    input['img_shape'] = (480, 640)
    input['ori_shape'] = (480, 640)
    input['pad_shape'] = (480, 640)
    input['cam2img'] = np.array([[[586.78313578, 0., 330.9516682, 0.],
                            [0., 586.32284279, 247.78813226, 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]]])
    input['lidar2cam'] = np.array([[[-0.00815028, -0.9998603 , -0.01459408,  0.03286563],
                            [-0.06233658,  0.0150742 , -0.9979413 , -0.1274831 ],
                            [ 0.9980219 , -0.00722376, -0.06245073, -0.1038875 ],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]]])
    input['box_type_3d'] = LiDARInstance3DBoxes
    input['box_mode_3d'] = Box3DMode.LIDAR
    # x, y, z, qx, qy, qz, qw = pose
    # quaternion = np.array([qw, qx, qy, qz])
    ego2global = np.eye(4)
    # ego2global[:3, :3] = Quaternion(quaternion).rotation_matrix
    # ego2global[:3, 3] = np.array([x, y, z])
    global2ego = np.linalg.inv(ego2global)
    
    # detection
    with torch.no_grad():
        for i in range(2, len(pipeline)):
            input = pipeline[i](input)
        input['data_samples'] = [input['data_samples']]
        input['inputs']['points'] = [input['inputs']['points']]
        input['inputs']['img'] = [input['inputs']['img']]
        output = detect_model.data_preprocessor(input, training=False)
        output = detect_model(**output, mode='predict')
        bboxes_3d = output[0].get('pred_instances_3d')['bboxes_3d']
        labels_3d = output[0].get('pred_instances_3d')['labels_3d']
        scores_3d = output[0].get('pred_instances_3d')['scores_3d']
        bboxes_3d = bboxes_3d[scores_3d > 0.3].tensor.cpu().numpy()
        labels_3d = labels_3d[scores_3d > 0.3].cpu().numpy()
        scores_3d = scores_3d[scores_3d > 0.3].cpu().numpy()
        
    # if process.get(count, None) is not None:
    #     xys = process[count]
    #     mask = []
    #     for i, bbox in enumerate(bboxes_3d):
    #         for xy in xys:
    #             if labels_3d[i] != xy[2] and (bbox[0]-xy[0])**2 + (bbox[1]-xy[1])**2 < 0.04:
    #                 break
    #         else:
    #             mask.append(i)
    #     bboxes_3d = bboxes_3d[mask]
    #     labels_3d = labels_3d[mask]
    #     scores_3d = scores_3d[mask]
    mask = labels_3d == 0
    bboxes_3d = bboxes_3d[mask]
    labels_3d = labels_3d[mask]
    scores_3d = scores_3d[mask]

    # tracking
    track_labels = []
    track_ids = []
    track_bboxes = []
    track_states = []
    for i, label in enumerate(class_labels):
        mask = labels_3d == label
        dets = np.concatenate([bboxes_3d[mask], scores_3d[mask][:, None]], axis=1).tolist()
        frame_data = FrameData(dets=dets, ego=ego2global, pc=None, det_types=labels_3d[mask], time_stamp=count/10)
        frame_data.dets = [BBox.bbox2world(ego2global, det) for det in frame_data.dets]
        results = trackers[i].frame_mot(frame_data)
        track_labels.append([trk[3] for trk in results])
        track_ids.append([trk[1] for trk in results])
        track_bboxes.append(np.array([BBox.bbox2array(trk[0]) for trk in results]))
        track_states.append([trk[2] for trk in results])

    # trajectory prediction
    topK = 3
    update_labels = []
    update_ids = []
    update_xys = []
    update_bboxes = []
    for i, label in enumerate(class_labels):
        for j in range(len(track_bboxes[i])):
            state = track_states[i][j].split('_')
            if state[0] == 'birth' or (state[0] == 'alive' and track_bboxes[i][j][-1] > 0.0005):
                update_labels.append(label)
                update_ids.append(track_ids[i][j])
                update_xys.append(track_bboxes[i][j][:2])
                update_bboxes.append(track_bboxes[i][j])
    update_labels.append(2)
    update_ids.append(0)
    update_xys.append(ego2global[:2, 3])
    update_bboxes.append(np.zeros(8))
    update_tracks(tracks, update_labels, update_ids, update_xys, update_bboxes, traj_pred_config)
    for i in range(len(update_labels)-1):
        label_id = str(update_labels[i]) + '_' + str(update_ids[i])
        now_track = tracks[label_id]
        if now_track['data'][-2][0] < 1e8:
            dir_x = now_track['data'][-1][0] - now_track['data'][-2][0]
            dir_y = now_track['data'][-1][1] - now_track['data'][-2][1]
            yaw = np.arctan2(dir_y, dir_x)
            update_bboxes[i][6] = yaw
            # if dir_y ** 2 + dir_x ** 2 > 0.25:
            #     update_labels[i] = 1      
    data_input = data_preprocess(tracks, traj_pred_config)
    if data_input is not None:
        with torch.no_grad():
            data_input = [tensor.cuda() for tensor in data_input]
            obs, neis, nei_masks, self_labels, nei_labels, refs, rot_mats, obs_bboxes = data_input
            obs = smooth_trajectories(obs)
            preds, scores, _ = traj_pred_model(obs, neis, nei_masks, self_labels, nei_labels)
            scores = torch.nn.functional.softmax(scores, dim=-1)
            topK_scores, topK_indices = torch.topk(scores, topK, dim=-1) # [B topK], [B topK]
            topK_preds = torch.gather(preds, 1, topK_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, preds.size(-2), preds.size(-1))) # [B topK pred_len in_size]
            rot_mats_T = rot_mats.transpose(1, 2)
            obs_ori = torch.matmul(obs, rot_mats_T) + refs.unsqueeze(1)
            preds_ori = torch.matmul(topK_preds, rot_mats_T.unsqueeze(1)) + refs.unsqueeze(1).unsqueeze(2)
            preds_ori = smooth_trajectories(preds_ori)
            obs_ori = obs_ori.cpu().numpy()
            preds_ori = preds_ori.cpu().numpy()
            self_labels = self_labels - 1

    # visualization
    vis.clear_geometries()
    # pointcloud
    pcd = o3d.geometry.PointCloud()
    points = input['inputs']['points'][0][:, :3].cpu().numpy()
    # points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    # points = np.dot(ego2global, points.T).T[:, :3]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 1, 1])
    # bboxes
    bboxes = np.stack([BBox.bbox2array(BBox.bbox2world(global2ego, BBox.array2bbox(bbox))) for bbox in update_bboxes])
    for i, bbox in enumerate(bboxes):
        if i == len(update_bboxes) - 1 :
        # if i == len(update_bboxes) - 1 or update_labels[i] == 1:
            continue
        color = label_colors[update_labels[i]]
        bbox[2] = bbox[2] + bbox[5] / 2
        bbox_geometry = o3d.geometry.OrientedBoundingBox(center=bbox[:3], extent=bbox[3:6], 
                        R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6])))
        pcd_in_box = bbox_geometry.get_point_indices_within_bounding_box(pcd.points)
        np.asarray(pcd.colors)[pcd_in_box] = np.array(color)
        bbox_geometry.color = color
        vis.add_geometry(bbox_geometry)
    vis.add_geometry(pcd)
    # trajectories
    if data_input is not None:
        for i in range(len(obs_ori)):
            # if self_labels[i] == 1:
            #     continue
            h = obs_bboxes[i][2].cpu().numpy()
            obs_points = np.concatenate([obs_ori[i], np.ones((obs_ori.shape[1], 1)) * h], axis=1)
            obs_points = np.dot(global2ego, np.concatenate([obs_points, np.ones((obs_points.shape[0], 1))], axis=1).T).T[:, :3]
            lines = [[j, j+1] for j in range(obs_ori.shape[1]-1)]
            line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(obs_points), lines=o3d.utility.Vector2iVector(lines))
            line_set.colors = o3d.utility.Vector3dVector([label_colors[self_labels[i]]] * len(lines))
            vis.add_geometry(line_set)
            for j in range(topK):
                pred_points = np.concatenate([preds_ori[i, j], np.ones((preds_ori.shape[2], 1)) * h], axis=1)
                pred_points = np.dot(global2ego, np.concatenate([pred_points, np.ones((pred_points.shape[0], 1))], axis=1).T).T[:, :3]
                pred_points = np.concatenate([obs_points[-1].reshape(1, 3), pred_points], axis=0)
                lines = [[j, j+1] for j in range(preds_ori.shape[2])]
                line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pred_points), lines=o3d.utility.Vector2iVector(lines))
                line_set.colors = o3d.utility.Vector3dVector([[0, 1, 1]] * len(lines))
                vis.add_geometry(line_set)
    
    infer_time += time.time() - start_time

    # camera view
    # ctr.set_lookat(ego2global[:3, 3])
    # ctr.set_zoom(0.5)
    # yaw = np.arctan2(ego2global[1, 0], ego2global[0, 0])
    # cam_dir = [-np.cos(yaw), -np.sin(yaw)] / np.linalg.norm([-np.cos(yaw), -np.sin(yaw)])
    # ctr.set_front([cam_dir[0], cam_dir[1], 1.5])
    # ctr.set_up([0, 0, 1])
    ctr.set_lookat([5, 0, 0])
    ctr.set_zoom(0.5)
    ctr.set_front([-1, 0, 1])
    ctr.set_up([0, 0, 1])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image('./infer/results/%04d.png' % count)  
    # with open('./infer/txt_results/%04d.json' % count, 'w', encoding='utf-8') as f:
    #     json.dump({'bboxes': bboxes_3d.tolist(), 'labels': labels_3d.tolist(), 'scores':scores_3d.tolist()}, f, ensure_ascii=False, indent=4)
    
    count += 1
vis.destroy_window()
print('infer time: %.4f' % infer_time)