import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, num_class, in_size, obs_len, pred_len, embed_size, pred_single):
        super(Encoder, self).__init__()
        self.num_class = num_class
        self.in_size = in_size
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.pred_single = pred_single
        if pred_single:
            self.self_embedding = nn.Linear(in_size*(obs_len+pred_len), embed_size)
        else:
            self.self_embedding = nn.ModuleList([nn.Linear(in_size*(obs_len+pred_len), embed_size) for _ in range(num_class)])
        self.nei_embedding = nn.ModuleList([nn.Linear(in_size*obs_len, embed_size) for _ in range(num_class + 1)])
        
    def forward(self, obs, neis, self_labels, nei_labels, init_trajs):
        '''
        Args:
            obs: [B obs_len in_size]
            neis: [B N obs_len in_size]
            self_labels: [B]
            nei_labels: [B N]
            init_trajs: [num_class num_init_trajs pred_len in_size]
        
        Return:
            x: [B num_init_trajs embed_size]
            nei_feats: [B N embed_size]
        '''
        B = obs.shape[0]
        N = neis.shape[1]
        
        if self.pred_single:
            init_trajs = init_trajs.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            init_trajs = init_trajs[self_labels]
        num_init_trajs = init_trajs.shape[1]

        # self embedding
        obs = obs.unsqueeze(1).repeat(1, num_init_trajs, 1, 1)
        traj = torch.cat((obs, init_trajs), dim=-2)
        traj = traj.reshape(B, num_init_trajs, -1)
        x = torch.zeros(B, num_init_trajs, self.embed_size).to(obs.device)
        if self.pred_single:
            x = self.self_embedding(traj)
        else:
            for i in range(self.num_class):
                mask = self_labels == i
                x[mask] = self.self_embedding[i](traj[mask])

        # nei embedding
        neis = neis.reshape(B, N, -1)
        nei_feats = torch.zeros(B, N, self.embed_size).to(obs.device)
        for i in range(self.num_class + 1):
            mask = nei_labels == i
            now_neis = neis[mask]
            positive = now_neis >= 0
            now_neis[positive] = 1 / (now_neis[positive] + 1e-4)
            now_neis[~positive] = 1 / (now_neis[~positive] - 1e-4)
            nei_feats[mask] = self.nei_embedding[i](now_neis)

        return x, nei_feats