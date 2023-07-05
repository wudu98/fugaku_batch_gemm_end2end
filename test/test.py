import time
import torch

print(torch.__version__)
dtype=torch.float32

MB = 1
P  = 128
H  = 7168
S  = 2048
NH = 56

MP = 1
repeats = 10

def run_layer(id, batch, M, N, K, repeats):
    print("Layer%d : batch-%d, M-%d, N-%d, K-%d" % (id, batch, M, N, K))
    Layer_input  = torch.randn(batch, M, K, dtype=dtype)
    Layer_wight  = torch.randn(batch, K, N, dtype=dtype)
    Layer_output_mm = torch.zeros(batch, M, N, dtype=dtype)
    Layer_output_bmm = torch.zeros(batch, M, N, dtype=dtype)

    start = time.time()
    for _ in range(repeats):
        Layer_output_bmm = torch.bmm(Layer_input, Layer_wight)
    end = time.time()

    latency = (end - start) / repeats
    flops = 2.0 * batch * M * N * K / latency * 1.e-9
    print("Layer%d torch.bmm(): %.4f ms, %.2f" % (id, latency * 1000, flops))

    start = time.time()
    for _ in range(repeats):
        for i in range(batch):
            Layer_output_mm[i,:,:] = torch.mm(Layer_input[i,:,:], Layer_wight[i,:,:])
    end = time.time()

    latency = (end - start) / repeats
    flops = 2.0 * batch * M * N * K / latency * 1.e-9
    print("Layer%d for torch.mm(): %.4f ms, %.2f" % (id, latency * 1000, flops))

    print("error_check_Layer%d:" % (id), torch.equal(Layer_output_bmm, Layer_output_mm))

def run_total(MB, P, H, S, NH, MP, repeats):
    # Layer 1
    batch=(int)(MB)
    M=(int)(S)
    N=(int)(P * NH * 3 / MP)
    K=(int)(H)
    run_layer(1, batch, M, N, K, repeats)

    # Layer 2
    batch=(int)(MB * NH / MP)
    M=(int)(S)
    N=(int)(S)
    K=(int)(P)
    run_layer(2, batch, M, N, K, repeats)

    # # Layer 3
    batch=(int)(MB * NH / MP)
    M=(int)(S)
    N=(int)(P)
    K=(int)(S)
    run_layer(3, batch, M, N, K, repeats)

    # # Layer 4
    batch=(int)(MB)
    M=(int)(S)
    N=(int)(H)
    K=(int)(H / MP)
    run_layer(4, batch, M, N, K, repeats)

    # # Layer 5
    batch=(int)(MB)
    M=(int)(S)
    N=(int)(4 * H / MP)
    K=(int)(H)
    run_layer(5, batch, M, N, K, repeats)

    # # Layer 6
    batch=(int)(MB)
    M=(int)(S)
    N=(int)(H)
    K=(int)(4 * H / MP)
    run_layer(6, batch, M, N, K, repeats)

run_total(MB, P, H, S, NH, MP, repeats)
