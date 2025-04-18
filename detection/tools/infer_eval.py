import os
import os.path as osp
import sys
sys.path.append(osp.abspath('./'))
import time
import numpy as np
import torch
import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.logging import print_log

save_path = 'test/work_dirs/'
cfg = Config.fromfile('my_projects/CMDT/configs/cmdt_coda.py')
cfg.work_dir = osp.abspath('./test/work_dirs')
test_info = 'coda_infos_test.pkl'
cfg.test_dataloader.dataset.ann_file = test_info
cfg.test_evaluator.ann_file = cfg.data_root + test_info
runner = Runner.from_cfg(cfg)
runner.load_checkpoint('ckpts/CMDT/cmdt_coda.pth')

sample_idxs = []
pre_bboxes = []
pre_labels = []
pre_scores = []
runner.model.eval()
count = 0
with torch.no_grad():
    for data_batch in runner.test_dataloader:
        data_batch = runner.model.data_preprocessor(data_batch, training=False)
        if isinstance(data_batch, dict):
            outputs = runner.model(**data_batch, mode='predict')
        elif isinstance(data_batch, (list, tuple)):
            outputs = runner.model(**data_batch, mode='predict')
        else:
            raise TypeError()
        sample_idxs.append(outputs[0].get('sample_idx'))
        bboxes = outputs[0].get('pred_instances_3d')['bboxes_3d'].tensor.numpy().copy()
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 5] / 2
        pre_bboxes.append(bboxes)
        pre_labels.append(outputs[0].get('pred_instances_3d')['labels_3d'].numpy())
        pre_scores.append(outputs[0].get('pred_instances_3d')['scores_3d'].numpy())
        runner.test_evaluator.process(data_samples=outputs, data_batch=data_batch)   
        count += 1
        if count % 100 == 0:
            print_log(f'Processed {count} / {len(runner.test_dataloader.dataset)} samples', logger='current')
    metrics = runner.test_evaluator.evaluate(len(runner.test_dataloader.dataset))

results = {
    'sample_idx': sample_idxs,
    'pre_bboxes': pre_bboxes,
    'pre_labels': pre_labels,
    'pre_scores': pre_scores
}
mmengine.utils.mkdir_or_exist(save_path)
mmengine.dump(results, save_path + 'coda_cmdt_detect_results.pkl')