import torch
import torch.nn as nn
from .trajectory_encoder import Encoder
from .trajectory_decoder import Decoder


class TrajectoryModel(nn.Module):

    def __init__(self, num_class, in_size, obs_len, pred_len, 
                 embed_size, num_decode_layers, num_modes):
        super(TrajectoryModel, self).__init__()
        self.num_class = num_class
        self.num_modes = num_modes
        self.encoder = Encoder(num_class, in_size, obs_len, embed_size)
        self.decoder = Decoder(num_class, in_size, pred_len, embed_size, num_decode_layers)
        self.modes = nn.Parameter(torch.empty(num_class, num_modes, embed_size))
        nn.init.uniform_(self.modes, 0, 1)

    def forward(self, obs, neis, nei_masks, self_labels, nei_labels):
        '''
        Args:
            obs: [B obs_len in_size]
            neis: [B N obs_len in_size]
            nei_masks: [B N]
            self_labels: [B]
            nei_labels: [B N]
            num_k: int

        Return: 
            pred [B num_modes pred_len in_size]
            scores [B num_modes]
        '''
        x, nei_feats = self.encoder(obs, neis, self_labels, nei_labels, self.modes)
        preds, scores = self.decoder(x, nei_feats, nei_masks, self_labels)
        return preds, scores