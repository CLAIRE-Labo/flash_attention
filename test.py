import unittest
import torch

from flash_mha import MultiheadFlashAttention
from mha import MultiHeadAttention
import common


class TestFlashAttention(unittest.TestCase):
    def test_equivalence(self):
        # Parameters
        seq_len = 64
        batch = 10
        num_heads = 4
        d_model = 100
        block_size = 16
        device = common.DEVICE
        test_data = 0.01 * torch.randn(batch, seq_len, d_model).to(device)
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads).to(device)
        flash_mha = MultiheadFlashAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size
        ).to(device)
        for param1, param2 in zip(mha.parameters(), flash_mha.parameters()):
            param2.data = param1.data 
        # Test if they do actually have the same parameters:
        for param1, param2 in zip(mha.parameters(), flash_mha.parameters()):
            if (param2.data != param1.data).all():
                raise ValueError("Two modules have different parameters!") 

        flash_out = flash_mha(test_data)
        mha_out = mha(test_data)
        self.assertTrue(flash_out.shape == mha_out.shape)
        torch.testing.assert_close(flash_out, mha_out)


if __name__ == "__main__":
    unittest.main()
