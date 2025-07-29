import torch
import torch.nn as nn
import time

from layers import SConv2d, SLinear


def benchmark_layer(layer, input_tensor, device, warmup=20, runs=50):
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = layer(input_tensor)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1e-3)
            else:
                t0 = time.time()
                _ = layer(input_tensor)
                t1 = time.time()
                times.append(t1 - t0)
    return sum(times) / len(times)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Conv2d speed test
    # standard_layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
    # standard_layer.to(device)
    # test_layer = SConv2d(1, 3, 3, 1, 1, 1, True)
    # test_layer.to(device)
    # input_tensor = torch.randn(32, 1, 28, 28)
    # input_tensor = input_tensor.to(device)
    #
    # std_time = benchmark_layer(standard_layer, input_tensor, device, warmup=20, runs=50)
    # test_time = benchmark_layer(test_layer, input_tensor, device, warmup=20, runs=50)
    #
    # print(f"nn.Conv2d average time: {std_time:.6f} s")
    # print(f"SConv2d   average time: {test_time:.6f} s")

    # Linear speed test
    standard_layer = nn.Linear(27, 81, bias=True)
    standard_layer.to(device)
    test_layer = SLinear(27, 81, maj_dim=3, bias=True)
    test_layer.to(device)
    input_tensor = torch.randn(32, 27, device=device)
    std_time = benchmark_layer(standard_layer, input_tensor, device, warmup=20, runs=50)
    test_time = benchmark_layer(test_layer, input_tensor, device, warmup=20, runs=50)
    print(f"nn.Linear average time: {std_time:.6f} s")
    print(f"SLinear   average time: {test_time:.6f} s")