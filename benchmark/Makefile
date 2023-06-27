CC=fcc
CCFLAG=-std=gnu11 -Nclang -Kfast,ocl,openmp -fPIC
BLAS=-SSL2BLAMP -lm -lfjomp

SRC=batch_gemm_benchmark.c
EXE=batch_gemm_benchmark

all : benchmark

benchmark :
	$(CC) $(SRC) $(CCFLAG) $(BLAS) -o $(EXE)

clean :
	rm -f $(EXE) *.o