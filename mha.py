import math
from einops import einsum
import logging
logging.basicConfig(level=logging.INFO)


import torch
from torch import nn
import common


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = None,
        num_heads: int = None,
        proj: nn.Module = None,
        out_proj: nn.ModuleDict = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by the num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scaling = 1.0 / math.sqrt(self.d_head)
        self.out_proj = out_proj if out_proj else nn.Linear(d_model, d_model)
        self.proj = proj if proj else nn.Linear(d_model, 3 * d_model, bias=False)

    def scaled_dot_product_attention(self, qkv: common.QKV, mask: torch.tensor = None):
        score = (
            einsum(
                qkv.Q,
                qkv.K,
                "b q_length n d, b kv_length n dim -> b n q_length kv_length",
            ) * self.scaling
        )
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        softmax_scores = torch.softmax(score, dim=-1)
        output = einsum(
            softmax_scores,
            qkv.V,
            "b n q_length kv_length, b kv_length n d -> b q_length n d",
        )
        return output

    def combine_heads(self, x):
        batch_size, seq_len, *_ = x.size()
        return x.contiguous().view(batch_size, seq_len, -1)

    def forward(self, x: torch.tensor, mask: torch.tensor = None):
        # Q, K, V: [batch_size, seq_len, d_model]
        QKV_proj = self.proj(x)
        Q, K, V = torch.split(QKV_proj, self.d_model, dim=-1)
        # Q, K, V map them to [batch_size, seq_len, num_heads, head_dim]
        Q, K, V = map(
            lambda x: torch.Tensor.view(x, (Q.size(0), Q.size(1), self.num_heads, -1)),
            (Q, K, V),
        )
        qkv = common.QKV(Q=Q, K=K, V=V)
        attention_out = self.scaled_dot_product_attention(qkv, mask=mask)
        output = self.out_proj(self.combine_heads(attention_out))
        return output