import unittest
import torch
import math
import sys
import torch.nn as nn
from unittest.mock import patch, MagicMock
from Scripts.embedding_blocks import SinusoidalPositionalEmbedding
sys.path.append('..')


class TestSinusoidalEmbeddings(unittest.TestCase):

    def test_positional_encodings(self):
        time_embed_dim = 10
        T=150
        penc = SinusoidalPositionalEmbedding(time_embed_dim)
        t = torch.randint(0, T, (128,1))
        out = penc(t)

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len=150):
                super().__init__()
                position = torch.arange(max_len).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(max_len, d_model)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)

            def forward(self, t):
                return self.pe[t.squeeze()]
            
        actual_emb = PositionalEncoding(time_embed_dim, max_len=T)(t)

        self.assertTrue(torch.allclose(actual_emb, out, atol=1e-5))