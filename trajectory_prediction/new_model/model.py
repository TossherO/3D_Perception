import torch
import torch.nn as nn
from .trajectory_encoder import Encoder
from .trajectory_decoder import Decoder


def generate_init_trajs(in_size, pred_len, scale, num_class=1):
    assert in_size == 2
    modes = [{'vel': 0, 'angle': 0.0}, {'vel': 1, 'angle': 0.0}, {'vel': 2, 'angle': 0.0},
             {'vel': 1, 'angle': 0.25}, {'vel': 1, 'angle': -0.25},
             {'vel': 0.5, 'angle': 0.5}, {'vel': 0.5, 'angle': -0.5}]
    num_modes = len(modes)
    init_trajs = torch.zeros(num_class, num_modes ** 2, pred_len, in_size, dtype=torch.float32)
    ori_pos = torch.zeros(in_size, dtype=torch.float32)
    ori_angle = torch.tensor(torch.pi, dtype=torch.float32)
    
    for i in range(num_class):
        for j, mode1 in enumerate(modes):
            for k, mode2 in enumerate(modes):
                traj = torch.zeros(pred_len, in_size, dtype=torch.float32)
                pos = ori_pos.clone()
                angle = ori_angle.clone()
                for t in range(pred_len):
                    vel = scale[i] * (mode1['vel'] if t < pred_len // 2 else mode2['vel'])
                    angle += torch.tensor(torch.pi, dtype=torch.float32) / pred_len / 2 * (mode1['angle'] if t < pred_len // 2 else mode2['angle'])
                    dir = torch.tensor([torch.cos(angle), torch.sin(angle)], dtype=torch.float32)
                    pos += vel * dir
                    traj[t] = pos
                init_trajs[i, j * num_modes + k] = traj
    
    if num_class == 1:
        init_trajs = init_trajs.squeeze(0)
    
    return init_trajs

    
class TrajectoryModel(nn.Module):

    def __init__(self, num_class, in_size, obs_len, pred_len, embed_size, num_decode_layers, 
                 scale, pred_single=False, init_trajs=None):
        super(TrajectoryModel, self).__init__()
        self.num_class = num_class
        self.encoder = Encoder(num_class, in_size, obs_len, pred_len, embed_size, pred_single)
        self.decoder = Decoder(num_class, in_size, pred_len, embed_size, num_decode_layers, pred_single)
        
        if init_trajs is not None:
            self.init_trajs = nn.Parameter(init_trajs)
        else:
            if pred_single:
                init_trajs = generate_init_trajs(in_size, pred_len, scale)
                self.init_trajs = nn.Parameter(init_trajs)
            else:
                init_trajs = generate_init_trajs(in_size, pred_len, scale, num_class)
                self.init_trajs = nn.Parameter(init_trajs)

    def forward(self, obs, neis, nei_masks, self_labels, nei_labels):
        '''
        Args:
            obs: [B obs_len in_size]
            neis: [B N obs_len in_size]
            nei_masks: [B N]
            self_labels: [B]
            nei_labels: [B N]

        Return: 
            pred [B num_init_trajs pred_len in_size]
            scores [B num_init_trajs]
            init_traj [B num_init_trajs pred_len in_size]
        '''
        x, nei_feats = self.encoder(obs, neis, self_labels, nei_labels, self.init_trajs)
        preds, scores = self.decoder(x, nei_feats, nei_masks, self_labels)
        return preds, scores, self.init_trajs[self_labels]