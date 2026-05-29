import torch
import torch.nn as nn
import torch.nn.functional as func


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.W_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.W_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.embed_dim = embed_dim


    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        k_T = k.transpose(-2,-1)

        similarity  = torch.matmul(q, k_T)

        scaled_sim  = similarity /(self.embed_dim ** 0.5)

        attention_perc = func.softmax(scaled_sim, dim=-1)

        return torch.matmul(attention_perc, v)
        





