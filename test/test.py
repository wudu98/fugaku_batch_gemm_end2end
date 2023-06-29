import time
import torch

print(torch.__version__)
dtype=torch.float32

loop = 3
P  = 128
H  = 7168
S  = 2048
NH = 56
MB = 48
MP = 4

# Layer 1
batch=(int)(MB)
M=(int)(S)
N=(int)(P * NH * 3 / MP)
K=(int)(H)

print("Layer1 : batch-%d, M-%d, N-%d, K-%d" % (batch, M, N, K))
Layer1_input  = torch.randn(batch, M, K, dtype=dtype)
Layer1_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer1_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer1_output = torch.bmm(Layer1_input, Layer1_wight)
end = time.time()

latency = (end - start) / loop
flops = 2.0 * batch * M * N * K / latency * 1.e-9
print("Layer1 : %.4f ms, %.2f" % (latency * 1000, flops))

# Layer 2
batch=(int)(MB * NH / MP)
M=(int)(S)
N=(int)(S)
K=(int)(P)

print("Layer2 : batch-%d, M-%d, N-%d, K-%d" % (batch, M, N, K))
Layer2_input  = torch.randn(batch, M, K, dtype=dtype)
Layer2_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer2_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer2_output = torch.bmm(Layer2_input, Layer2_wight)
end = time.time()

latency = (end - start) / loop
flops = 2.0 * batch * M * N * K / latency * 1.e-9
print("Layer2 : %.4f ms, %.2f" % (latency * 1000, flops))

# Layer 3
batch=(int)(MB * NH / MP)
M=(int)(S)
N=(int)(P)
K=(int)(S)

print("Layer3 : batch-%d, M-%d, N-%d, K-%d" % (batch, M, N, K))
Layer3_input  = torch.randn(batch, M, K, dtype=dtype)
Layer3_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer3_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer3_output = torch.bmm(Layer3_input, Layer3_wight)
end = time.time()

latency = (end - start) / loop
flops = 2.0 * batch * M * N * K / latency * 1.e-9
print("Layer3 : %.4f ms, %.2f" % (latency * 1000, flops))

# Layer 4
batch=(int)(MB)
M=(int)(S)
N=(int)(H)
K=(int)(H / MP)

print("Layer4 : batch-%d, M-%d, N-%d, K-%d" % (batch, M, N, K))
Layer4_input  = torch.randn(batch, M, K, dtype=dtype)
Layer4_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer4_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer4_output = torch.bmm(Layer4_input, Layer4_wight)
end = time.time()

latency = (end - start) / loop
flops = 2.0 * batch * M * N * K / latency * 1.e-9
print("Layer4 : %.4f ms, %.2f" % (latency * 1000, flops))

# Layer 5
batch=(int)(MB)
M=(int)(S)
N=(int)(4 * H / MP)
K=(int)(H)

print("Layer5 : batch-%d, M-%d, N-%d, K-%d" % (batch, M, N, K))
Layer5_input  = torch.randn(batch, M, K, dtype=dtype)
Layer5_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer5_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer5_output = torch.bmm(Layer5_input, Layer5_wight)
end = time.time()

latency = (end - start) / loop
flops = 2.0 * batch * M * N * K / latency * 1.e-9
print("Layer5 : %.4f ms, %.2f" % (latency * 1000, flops))

# Layer 6
batch=(int)(MB)
M=(int)(S)
N=(int)(H)
K=(int)(4 * H / MP)

print("Layer6 : batch-%d, M-%d, N-%d, K-%d" % (batch, M, N, K))
Layer6_input  = torch.randn(batch, M, K, dtype=dtype)
Layer6_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer6_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer6_output = torch.bmm(Layer6_input, Layer6_wight)
end = time.time()

latency = (end - start) / loop
flops = 2.0 * batch * M * N * K / latency * 1.e-9
print("Layer6 : %.4f ms, %.2f" % (latency * 1000, flops))