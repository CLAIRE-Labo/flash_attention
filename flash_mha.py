import math
from einops import einsum, rearrange
import logging
logging.basicConfig(level=logging.INFO)


import torch
from torch import nn
import common


class FlashAttention(nn.Module):
    def __init__(self, block_size=64):
        super().__init__()
        self.block_size = block_size

    def forward(self, qkv: common.QKV, mask=None):
        O = torch.zeros_like(qkv.Q, requires_grad=True)
        l = torch.zeros(qkv.Q.shape[:-1])
        m = torch.ones(qkv.Q.shape[:-1]) * common.MASKOUT_VAL
        device = qkv.Q.device

        O, l, m = map(lambda x: x.to(device), (O, l, m))
        q_seqlen = qkv.Q.shape[1]
        kv_seqlen = qkv.K.shape[1]
        Q_block_size = min(self.block_size, q_seqlen)
        KV_block_size = min(self.block_size, kv_seqlen)

        Q_blocks = torch.split(qkv.Q, Q_block_size, dim=1)
        K_blocks = torch.split(qkv.K, KV_block_size, dim=1)
        V_blocks = torch.split(qkv.V, KV_block_size, dim=1)
        if mask:
            mask_blocks = torch.split(mask, KV_block_size, dim=1)

        scale = 1.0 / math.sqrt(qkv.Q.shape[-1])
        O_blocks = list(torch.split(O, Q_block_size, dim=1))
        l_blocks = list(torch.split(l, Q_block_size, dim=1))
        m_blocks = list(torch.split(m, Q_block_size, dim=1))
        
        num_q_blocks = len(Q_blocks)
        num_kv_blocks = len(K_blocks)
        
        print(f"num_q_blocks: {num_q_blocks}, num_kv_blocks: {num_kv_blocks}")
        for j in range(num_kv_blocks):
            K_block_j, V_block_j = K_blocks[j], V_blocks[j]
            if mask:
                mask_block_j = mask_blocks[..., j]

            for i in range(num_q_blocks):
                print(f"i: {i} j: {j}")
                Q_block_i = Q_blocks[i]

                if j == 0:
                    O_block_i = rearrange(O_blocks[i], "b l n d -> b n l d")
                    m_i = rearrange(m_blocks[i], "b l n -> b n l 1")
                    l_i = rearrange(l_blocks[i], "b l n -> b n l 1")

                S_ij = (
                    einsum(
                        Q_block_i,
                        K_block_j,
                        "b block_len_i n d, b block_len_j n d -> b n block_len_i block_len_j",
                    )
                    * scale
                )

                if mask:
                    S_ij = S_ij.masked_fill(mask_block_j != 0, S_ij, common.MASKOUT_VAL)

                m_ij = torch.max(S_ij, dim=-1, keepdim=True).values
                P_ij = torch.exp(S_ij - m_ij)
                l_ij = P_ij.sum(-1, keepdim=True)
                m_i_new = torch.maximum(m_i, m_ij)
                l_i_new = (
                    torch.exp(m_i - m_i_new) * l_i + torch.exp(m_ij - m_i_new) * l_ij
                )
                O_block_i_new = einsum(
                    P_ij,
                    V_block_j,
                    "b n q_length kv_length, b kv_length n d -> b n q_length d",
                )
                O_blocks[i] = (l_i * torch.exp(m_i - m_i_new) * O_block_i + 
                    torch.exp(m_ij - m_i_new) * O_block_i_new) / l_i_new
                l_blocks[i] = l_i_new.squeeze(-1)
                m_blocks[i] = m_i_new.squeeze(-1)
        attn_output = rearrange(torch.cat(O_blocks, dim=2), "b n l d -> b l n d")
        return attn_output


class MultiheadFlashAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        proj: nn.Module = None,
        out_proj: nn.Module = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by the num_heads."
        assert (
            block_size <= d_model
        ), "the block size must be less than or equal to d_model."
        self.attention = FlashAttention(block_size=block_size)
        self.d_model = d_model
        self.num_heads = num_heads
        self.out_proj = out_proj if out_proj else nn.Linear(d_model, d_model)
        self.proj = proj if proj else nn.Linear(d_model, 3 * d_model, bias=False)

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
        attention_out = self.attention(qkv, mask=mask)
        output = self.out_proj(self.combine_heads(attention_out))
        return output
