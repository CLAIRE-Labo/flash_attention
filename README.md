# flash_attention
A basic pure pytorch implementation of flash attention.

The codebase is mainly written for educational purposes not meant to be used for production
or anything serious. For more practical use cases consider using [Flexattention](https://pytorch.org/blog/flexattention/).

I would refer to the original paper for the details of the algorithm. This implementation
is based on the Algorithm 1 in the paper:

```
Dao, Tri, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. "Flashattention: Fast and memory-efficient exact attention with io-awareness." Advances in Neural Information Processing Systems 35 (2022): 16344-16359.
```

The original implementation requires CUDA kernels for fusing operations and moving data
between HBM to SRAM whereas this implementation does not consider any of these.

## Implementation
There are four important files:
- `flash_mha.py`: A very basic implementation of flash attention in pytorch. 
For educational purposes only.
- `mha.py`: A vanilla implementation of the multi-head attention.
- `test.py`: A unit test that checks the equivalence of the forward prop of flash attention
and the vanilla multihead attention.
- `benchmark.py`: For the speed and memory comparisons between the flash attention and 
the vanilla multihead attention.

To run the test you can simply run:

`python test.py`

To run the benchmark you can simply run:

`python benchmark.py`