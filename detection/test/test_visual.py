import os
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))
import torch
import numpy as np
import mmcv
import mmengine
import open3d as o3d
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet3d.utils import replace_ceph_backend
from mmengine.structures import InstanceData
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures import Box3DMode
from mmdet3d.registry import MODELS, TRANSFORMS

# cfg = Config.fromfile('my_projects/CMT/configs/cmt_nus.py')
cfg = Config.fromfile('my_projects/CMDT/configs/cmdt_coda.py')
# cfg = Config.fromfile('my_projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py')
cfg.work_dir = osp.abspath('./test/work_dirs')
runner = Runner.from_cfg(cfg)
visualizer1 = Det3DLocalVisualizer()
visualizer2 = Det3DLocalVisualizer()
visualizer3 = Det3DLocalVisualizer()
label_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

# runner.train_dataloader.dataset.dataset.pipeline.transforms.pop(3)
runner.load_checkpoint('ckpts/CMDT/cmdt_coda.pth')
runner.model.eval()

count = 2600
info_path = './data/CODA/coda_infos_val.pkl'
data_info = mmengine.load(info_path)
input = data_info['data_list'][count]
token = input['token']
timestamp = input['timestamp']
ego2global = input['ego2global']
path_prefix = info_path[:info_path.rfind('/')+1]
input['lidar_points']['lidar_path'] = path_prefix + input['lidar_points']['lidar_path']
for image in input['images'].values():
    image['img_path'] = path_prefix + image['img_path']
input['box_type_3d'] = LiDARInstance3DBoxes
input['box_mode_3d'] = Box3DMode.LIDAR
gt_bboxes = [instance['bbox_3d'] for instance in input['instances']]
gt_labels = [instance['bbox_label_3d'] for instance in input['instances']]
pipeline = []
for transform in cfg.test_dataloader.dataset.pipeline:
    pipeline.append(TRANSFORMS.build(transform))

# detection
with torch.no_grad():
    for transform in pipeline:
        input = transform(input)
    input['data_samples'] = [input['data_samples']]
    input['inputs']['points'] = [input['inputs']['points']]
    input['inputs']['img'] = [input['inputs']['img']]
# count = 2600
# for data_batch in runner.test_dataloader:
#     if count == 0:
#         break
#     count -= 1
data_batch = runner.model.data_preprocessor(input, training=False)
batch_inputs_dict = data_batch['inputs']
batch_data_samples = data_batch['data_samples']
imgs = batch_inputs_dict.get('imgs', None)
points = batch_inputs_dict.get('points', None)
img_metas = [item.metainfo for item in batch_data_samples]

# gt_bboxes_3d = [item.get('eval_ann_info')['gt_bboxes_3d'] for item in batch_data_samples]
# gt_labels_3d = [item.get('eval_ann_info')['gt_labels_3d'] for item in batch_data_samples]
# gt_bboxes_3d = [item.get('gt_instances_3d')['bboxes_3d'] for item in batch_data_samples]
# gt_labels_3d = [item.get('gt_instances_3d')['labels_3d'] for item in batch_data_samples]

img = imgs[0][0].permute(1, 2, 0).cpu().numpy()
img = mmcv.imdenormalize(img, mean=np.array([103.530, 116.280, 123.675]), std=np.array([57.375, 57.120, 58.395]), to_bgr=True)
point = points[0].cpu().numpy()
bboxes_3d = LiDARInstance3DBoxes(gt_bboxes, origin=(0.5, 0.5, 0.5))
bbox_color = [label_colors[label] for label in gt_labels]

# img_aug_matrix = img_metas[0]['img_aug_matrix'][0]
# cam2img = img_metas[0]['cam2img'][0]
# lidar2cam = img_metas[0]['lidar2cam'][0]
# cam2img[:2, :3] = img_aug_matrix[:2, :2] @ cam2img[:2, :3]
# cam2img[:2, 2] += img_aug_matrix[:2, 3]
# lidar2img = cam2img @ lidar2cam
lidar2img = img_metas[0]['lidar2img'][0]
input_meta = {'lidar2img': lidar2img}

if isinstance(data_batch, dict):
    outputs = runner.model(**data_batch, mode='predict')
elif isinstance(data_batch, (list, tuple)):
    outputs = runner.model(**data_batch, mode='predict')
else:
    raise TypeError()
runner.val_evaluator.process(data_samples=outputs, data_batch=data_batch)

bboxes_3d_pre = outputs[0].get('pred_instances_3d').get('bboxes_3d')
labels_3d_pre = outputs[0].get('pred_instances_3d').get('labels_3d')
scores_3d_pre = outputs[0].get('pred_instances_3d').get('scores_3d')
bboxes_3d_pre = bboxes_3d_pre[scores_3d_pre > 0.3]
labels_3d_pre = labels_3d_pre[scores_3d_pre > 0.3]
scores_3d_pre = scores_3d_pre[scores_3d_pre > 0.3]
bbox_color_pre = [label_colors[label] for label in labels_3d_pre]
print(scores_3d_pre)

visualizer1.set_points(point, points_size=1)
visualizer1.draw_bboxes_3d(bboxes_3d, bbox_color)
# visualizer1.draw_bboxes_3d(bboxes_3d_pre, bbox_color_pre)

visualizer2.set_image(img)
visualizer2.draw_proj_bboxes_3d(bboxes_3d, input_meta)
# print(img_metas[0]['img_path'])

visualizer3.set_points(point, points_size=1)
visualizer3.draw_bboxes_3d(bboxes_3d_pre, bbox_color_pre)

visualizer1.show()
visualizer2.show()
visualizer3.show()