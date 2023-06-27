import time
import torch

print(torch.__version__)
dtype=torch.float32

loop = 10
P  = 128
H  = 7168
S  = 2048
NH = 56
MB = 1
MP = 1

# Layer 1
batch=(int)(1)
M=(int)(S)
N=(int)(P * NH * 3 / MP)
K=(int)(H)

print("Layer1 : M %d, N %d, K %d" % (M, N, K))
Layer1_input  = torch.randn(batch, M, K, dtype=dtype)
Layer1_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer1_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer1_output = torch.bmm(Layer1_input, Layer1_wight)
end = time.time()

print("Layer1 : %.4f ms" % (1000 * (end - start) / loop))

# Layer 2
batch=(int)(NH / MP)
M=(int)(S)
N=(int)(S)
K=(int)(P)

print("Layer2 : M %d, N %d, K %d" % (M, N, K))
Layer2_input  = torch.randn(batch, M, K, dtype=dtype)
Layer2_wight  = torch.randn(batch, K, N, dtype=dtype)
Layer2_output = torch.zeros(batch, M, N, dtype=dtype)

start = time.time()
for i in range(loop):
    Layer2_output = torch.bmm(Layer2_input, Layer2_wight)
end = time.time()

print("Layer2 : %.4f ms" % (1000 * (end - start) / loop))