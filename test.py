import unittest
import torch

from flash_mha import MultiheadFlashAttention
from mha import MultiHeadAttention


class TestFlashAttention(unittest.TestCase):
    def test_equivalence(self):
        # Parameters
        seq_len = 64
        batch = 10
        num_heads = 4
        d_model = 100
        block_size = 16
        device = "cuda"
        test_data = torch.randn(batch, seq_len, d_model).to(device)
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
        flash_mha = MultiheadFlashAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            proj=mha.proj,
            out_proj=mha.out_proj
        ).to(device)
        flash_out = flash_mha(test_data)
        mha_out = mha(test_data)
        import pdb

        pdb.set_trace()
        self.assertTrue(flash_out.shape == mha_out.shape)
        self.assertTrue(torch.testing.assert_close(flash_out, mha_out))


if __name__ == "__main__":
    unittest.main()
