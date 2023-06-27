import time
import torch

print(torch.__version__)
dtype=torch.float32

ite = 1000
batch = 1
attention = 16
A = torch.randn(batch * attention, 384, 64, dtype=dtype)
B = torch.randn(batch * attention, 64, 384, dtype=dtype)

C = torch.bmm(A, B)

start = time.time()
for i in range(ite):
    res = torch.bmm(A, B)
end = time.time()

print((end - start)/ite)
