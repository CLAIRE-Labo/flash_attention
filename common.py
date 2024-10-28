from dataclasses import dataclass
import torch

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
    DEVICE = torch.device("cuda:0")
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    raise ValueError("GPU or CUDA is not available.")


MASKOUT_VAL = -float("inf")
EPS = 1e-10


@dataclass
class ModelParams:
    d_model: int
    num_heads: int
    block_size: int
    d_ffn: int


@dataclass
class QKV:
    Q: torch.tensor
    K: torch.tensor
    V: torch.tensor
