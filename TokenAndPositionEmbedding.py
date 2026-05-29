import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_dim, context_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)

        self.pos_embeddings = nn.Embedding(context_size, embed_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape

        positions = torch.arange(0,seq_len, device=x.device).unsqueeze(0)

        token_embeddings = self.token_embeddings(x)

        position_embeddings = self.pos_embeddings(positions)

        return token_embeddings + position_embeddings