import argparse
import time
import os
import torch
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO)

from flash_mha import MultiheadFlashAttention
from mha import MultiHeadAttention
import common

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    os.environ['USE_KINETO'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, required=False, type=int)
parser.add_argument('--seq_len', default=8192, required=False, type=int)
parser.add_argument('--d_model', default=1024, required=False, type=int)
parser.add_argument('--num_heads', default=16, required=False, type=int)
parser.add_argument('--block_size', default=1024, required=False, type=int)
parser.add_argument('--profile', default=True, required=False, type=bool)
args = parser.parse_args()


def time_attention(data: torch.tensor = None, 
                   num_trials: int = 10, 
                   attention_module = torch.nn.Module):
    
    ms_denom = 1_000_000
    device = common.DEVICE
    attention_module = attention_module.to(device)
    data = data.to(device)
    # Warmup:
    for _ in range(10):
        attention_module(data)
    # The real timing starts now:
    times = []
    for _ in range(num_trials):
        start = time.time_ns()
        attention_module(data)
        end = time.time_ns()
        duration = (end - start) / ms_denom
        times.append(duration)
    logging.warning(f"Average time: {np.mean(times)} ms, Std time: {np.std(times)}.")


def run_profile(data: torch.tensor = None, 
                num_trials: int = 10,
                attention_module = torch.nn.Module):
    device = common.DEVICE
    attention_module = attention_module.to(device)
    data = data.to(device)
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/tmp/profiler_logs/bench_log_flash'),
            record_shapes=True,
            profile_memory=True,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
        ) as prof:
            for _ in range(num_trials):
                attention_module(data)
    logging.warning(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

def run_benchmarks():
    data = torch.randn((args.batch_size, args.seq_len, args.d_model))
    mha_module = MultiHeadAttention(d_model=args.d_model, num_heads=args.num_heads)
    proj = mha_module.proj
    out_proj = mha_module.out_proj
    flash_mha_module = MultiheadFlashAttention(d_model=args.d_model, num_heads=args.num_heads, 
                                               block_size=args.block_size, proj=proj, out_proj=out_proj)
    logging.warning("Benchmarking flash multihead attention...")
    time_attention(data, attention_module=flash_mha_module)
    logging.warning("========")
    logging.warning("Benchmarking the regular multihead attention...")
    time_attention(data, attention_module=mha_module)
    logging.warning("Finished timing!")
    logging.warning("Profiling flash multihead attention...")
    run_profile(data, num_trials=20, attention_module=flash_mha_module)
    logging.warning("========")
    logging.warning("Profiling flash multihead attention...")
    run_profile(data, num_trials=20, attention_module=mha_module)
    logging.warning("Finished profiling!")


if __name__ == "__main__":
    run_benchmarks()

    