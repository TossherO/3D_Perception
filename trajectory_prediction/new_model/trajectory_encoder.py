import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, num_class, in_size, obs_len, embed_size, pred_single):
        super(Encoder, self).__init__()
        self.num_class = num_class
        self.in_size = in_size
        self.obs_len = obs_len
        self.embed_size = embed_size
        self.pred_single = pred_single
        if pred_single:
            self.obs_embedding = nn.Linear(in_size*obs_len, embed_size)
        else:
            self.obs_embedding = nn.ModuleList([nn.Linear(in_size*obs_len, embed_size) for _ in range(num_class)])
        self.nei_embedding = nn.ModuleList([nn.Linear(in_size*obs_len, embed_size) for _ in range(num_class + 1)])
        self.mode_embedding = nn.Linear(2*embed_size, embed_size)
        
    def forward(self, obs, neis, self_labels, nei_labels, modes):
        '''
        Args:
            obs: [B obs_len in_size]
            neis: [B N obs_len in_size]
            self_labels: [B]
            nei_labels: [B N]
            modes: [num_class num_modes embed_size]
        
        Return:
            x: [B num_modes embed_size]
            nei_feats: [B N embed_size]
        '''
        B = obs.shape[0]
        N = neis.shape[1]
        obs = obs.reshape(B, -1)
        neis = neis.reshape(B, N, -1)

        # trajectory embedding
        x = torch.zeros(B, self.embed_size).to(obs.device)
        nei_feats = torch.zeros(B, N, self.embed_size).to(obs.device)

        # obs embedding
        if self.pred_single:
            x = self.obs_embedding(obs)
        else:
            for i in range(self.num_class):
                mask = self_labels == i
                x[mask] = self.obs_embedding[i](obs[mask])

        # nei embedding
        for i in range(self.num_class + 1):
            mask = nei_labels == i
            now_neis = neis[mask]
            positive = now_neis >= 0
            now_neis[positive] = 1 / (now_neis[positive] + 1e-4)
            now_neis[~positive] = 1 / (now_neis[~positive] - 1e-4)
            nei_feats[mask] = self.nei_embedding[i](now_neis)

        # mode embedding
        if self.pred_single:
            mode_feats = modes.unsqueeze(0).repeat(B, 1, 1)
        else:
            mode_feats = modes[self_labels]
        x = x.unsqueeze(1).repeat(1, mode_feats.shape[1], 1)
        x = torch.cat((x, mode_feats), dim=-1)
        x = self.mode_embedding(x)

        return x, nei_feats