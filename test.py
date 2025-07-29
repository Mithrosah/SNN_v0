import torch
import torch.nn as nn
import torch.nn.functional as F




def f2s(seq_len, float_tensor: torch.Tensor) -> torch.Tensor:
    assert seq_len % 32 == 0, "seq_len should be multiple of 32"
    p = (float_tensor + 1) / 2
    dims = len(float_tensor.shape)
    p = p.unsqueeze(-1).expand(*(-1,) * dims, seq_len)
    bits = torch.bernoulli(p)
    bits = bits.view(*bits.shape[:-1], seq_len // 32, 32)
    weights = torch.tensor([1 << i for i in range(32)], dtype=torch.int64, device=bits.device)
    packed = (bits.int() * weights).sum(dim=-1)
    return packed



def s2f(seq_len, packed_stream: torch.Tensor) -> torch.Tensor:
    num_ints = packed_stream.shape[-1]
    assert (
        num_ints == seq_len // 32
    ), f"wrong number of ints, expect {seq_len // 32}, got {num_ints}"
    popcount = torch.zeros_like(packed_stream, dtype=torch.int32)
    for i in range(32):
        popcount += (packed_stream >> i) & 1
    total_ones = popcount.sum(dim=-1)
    p = total_ones / seq_len
    return p * 2 - 1
#
# packed = f2s(64, torch.tensor([[0.5, 0.0, -0.3], [0.0, 0.3, 0.1]]))
# f_num = s2f(64, packed)
# print(packed)
# print(f_num)