import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class Decoder(nn.Module):

    def __init__(self, num_class, in_size, pred_len, embed_size, num_layers, pred_single,
                 head=4, blocks=[['self', 'cross'], ['self', 'cross']]):
        super(Decoder, self).__init__()
        self.num_class = num_class
        self.in_size = in_size
        self.pred_len = pred_len
        self.pred_single = pred_single
        self.decoderLayers = nn.ModuleList([DecoderLayer(embed_size, head, blocks[0]) for _ in range(num_layers)])
        if pred_single:
            self.pred_head = nn.Linear(embed_size, in_size*pred_len)
            self.socre_head = nn.Linear(embed_size, 1)
        else:
            self.pred_head = nn.ModuleList([nn.Linear(embed_size, in_size*pred_len) for _ in range(num_class)])
            self.socre_head = nn.ModuleList([nn.Linear(embed_size, 1) for _ in range(num_class)])

    def forward(self, x, nei_feats, nei_masks, self_labels):
        '''
        Args:
            x: [B num_init_trajs embed_size]
            nei_feats: [B N embed_size]
            nei_masks: [B N]
            self_labels: [B]

        Return:
            pred [B num_init_trajs pred_len in_size]
            scores [B num_init_trajs]
        '''
        B = x.shape[0]
        K = x.shape[1]

        # transformer decoder
        for layer in self.decoderLayers:
            x = layer(x, nei_feats, nei_masks)
        
        # prediction and score
        preds = torch.zeros(B, K, self.pred_len, self.in_size).to(x.device)
        scores = torch.zeros(B, K).to(x.device)
        if self.pred_single:
            preds = self.pred_head(x).reshape(-1, K, self.pred_len, self.in_size)
            scores = self.socre_head(x).squeeze(-1)
        else:
            for i in range(self.num_class):
                mask = self_labels == i
                preds[mask] = self.pred_head[i](x[mask]).reshape(-1, K, self.pred_len, self.in_size)
                scores[mask] = self.socre_head[i](x[mask]).squeeze(-1)
        return preds, scores


class DecoderLayer(nn.Module):

    def __init__(self, embed_size, head, blocks):
        super(DecoderLayer, self).__init__()
        self.blocks = blocks
        self.attns = nn.ModuleList([TransformerBlock(embed_size, head, forward_expansion=2, 
                                                     dropout=0.1) for _ in range(len(blocks))])

    def forward(self, x, nei_feats, nei_masks):
        '''
        Args:
            x: [B num_init_trajs embed_size]
            nei_feats: [B N embed_size]
            nei_masks: [B N]

        Return:
            x [B num_init_trajs embed_size]
        '''
        for i, block in enumerate(self.blocks):
            if block == 'self':
                x = self.attns[i](x, x, x, mask=None)
            elif block == 'cross':
                masks = nei_masks.unsqueeze(1).repeat(1, x.shape[1], 1)  # [B K N]
                x = self.attns[i](x, nei_feats, nei_feats, masks)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, embed_size, head, forward_expansion, dropout, islinear=True):
        super(TransformerBlock, self).__init__()
        self.attn = MultihHeadAttention(embed_size, head, islinear=islinear)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.feed_forward = FeedForwardLayer(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        '''
        Args:
            query: [B K embed_size]
            key: [B N embed_size]
            value: [B N embed_size]
            mask: [B K N]

        Return:
            out [B K embed_size]
        '''
        logits = self.attn(query, key, value, mask)  # [B K embed_size]
        x = self.dropout(self.norm1(logits + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, forward_expansion):
        super(FeedForwardLayer, self).__init__()
        self.w1 = nn.Linear(d_model, d_model*forward_expansion)
        self.w2 = nn.Linear(d_model*forward_expansion, d_model)

    def forward(self, x):
        return self.w2((F.relu(self.w1(x))))


class LayerNorm(nn.Module):

    def __init__(self, embedding_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(embedding_dim))
        self.b = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x-mean) / (std+self.eps) + self.b


class MultihHeadAttention(nn.Module):

    def __init__(self, d_model, h, dropout=0.1, islinear=True):
        super(MultihHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_key = nn.Linear(d_model, d_model) if islinear else nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_query = nn.Linear(d_model, d_model) if islinear else nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.atten = None  

    def forward(self, query, key, value, mask=None):
        
        '''
        Args:
            query: [B K d_model]
            key: [B N d_model]
            value: [B N d_model]
            mask: [B K N]

        Return:
            logits [B K d_model]
        '''
        
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)  # [B h K N] adding the dimension of head

        query = self.w_query(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # [B h K d_k]
        key = self.w_key(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # [B h N d_k]
        value = self.w_value(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # [B h N d_k]
        
        x, self.atten = attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k) # [B K d_model]
        return self.fc_out(x)


def attention(query, key, value, mask=None, dropout=None):
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [B h K N]
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    scores = F.softmax(scores, dim=-1)  # [B h K N] 
    # scores = torch.tanh(scores)
    if dropout is not None:
        scores = dropout(scores)
    logits = torch.matmul(scores, value)  # [B h K d_k]
    return logits, scores