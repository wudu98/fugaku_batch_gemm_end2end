import time
import torch

print(torch.__version__)
dtype=torch.float32

loop = 3
P  = 128
H  = 7168
S  = 2048
NH = 56
MB = 1
MP = 1

def run_layer(id, batch, M, N, K):
    print("Layer%d : batch-%d, M-%d, N-%d, K-%d" % (id, batch, M, N, K))
    Layer_input  = torch.randn(batch, M, K, dtype=dtype)
    Layer_wight  = torch.randn(batch, K, N, dtype=dtype)
    Layer_output_mm = torch.zeros(batch, M, N, dtype=dtype)
    Layer_output_bmm = torch.zeros(batch, M, N, dtype=dtype)

    start = time.time()
    for i in range(loop):
        Layer_output_bmm = torch.bmm(Layer_input, Layer_wight)
    end = time.time()

    latency = (end - start) / loop
    flops = 2.0 * batch * M * N * K / latency * 1.e-9
    print("Layer%d torch.bmm(): %.4f ms, %.2f" % (id, latency * 1000, flops))

    start = time.time()
    for i in range(loop):
        for j in range(batch):
            Layer_output_mm[j,:,:] = torch.mm(Layer_input[j,:,:], Layer_wight[j,:,:])
    end = time.time()

    latency = (end - start) / loop
    flops = 2.0 * batch * M * N * K / latency * 1.e-9
    print("Layer1 for torch.mm(): %.4f ms, %.2f" % (latency * 1000, flops))

    print("error_check_Layer1:",torch.equal(Layer1_output_bmm, Layer1_output_mm))

# Layer 1
batch=(int)(MB)
M=(int)(S)
N=(int)(P * NH * 3 / MP)
K=(int)(H)
run_layer(1, batch, M, N, K)

# Layer 2
batch=(int)(MB * NH / MP)
M=(int)(S)
N=(int)(S)
K=(int)(P)
run_layer(2, batch, M, N, K)

# # Layer 3
batch=(int)(MB * NH / MP)
M=(int)(S)
N=(int)(P)
K=(int)(S)
run_layer(3, batch, M, N, K)

# # Layer 4
batch=(int)(MB)
M=(int)(S)
N=(int)(H)
K=(int)(H / MP)
run_layer(4, batch, M, N, K)

# # Layer 5
batch=(int)(MB)
M=(int)(S)
N=(int)(4 * H / MP)
K=(int)(H)
run_layer(5, batch, M, N, K)

# # Layer 6
batch=(int)(MB)
M=(int)(S)
N=(int)(H)
K=(int)(4 * H / MP)
run_layer(6, batch, M, N, K)
