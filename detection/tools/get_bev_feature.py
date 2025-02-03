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
cfg.model.return_pts_feat = True
runner = Runner.from_cfg(cfg)
runner.load_checkpoint('ckpts/CMDT/coda.pth')

sample_idxs = []
bev_features = []
runner.model.eval()
count = 0
with torch.no_grad():
    for data_batch in runner.test_dataloader:
        data_batch = runner.model.data_preprocessor(data_batch, training=False)
        if isinstance(data_batch, dict):
            outputs, bev_feature = runner.model(**data_batch, mode='predict')
        elif isinstance(data_batch, (list, tuple)):
            outputs, bev_feature = runner.model(**data_batch, mode='predict')
        else:
            raise TypeError()
        sample_idxs.append(outputs[0].get('sample_idx'))
        bev_features.append(bev_feature[0].cpu().numpy())
        count += 1
        if count % 100 == 0:
            print_log(f'Processed {count} / {len(runner.test_dataloader.dataset)} samples', logger='current')
        
bev_feature_result = {
    'sample_idx': sample_idxs,
    'bev_features': bev_features
}
mmengine.utils.mkdir_or_exist(save_path)
mmengine.dump(bev_feature_result, save_path + 'bev_features.pkl')