CC=fcc
CCFLAG=-Nclang -Kfast,ocl,openmp -fPIC -I/home/ra000012/a04083/gnn/gnn/wudu/LLM_speedup/benchmark/dl4fugaku/build/pytorch -I/home/ra000012/a04083/gnn/gnn/wudu/LLM_speedup/benchmark/dl4fugaku/build/pytorch/aten/src
BLAS=-SSL2BLAMP -lm -lfjomp

LIB_BATCHED_BLAS_SO = libbatch_sgemm_intercept.so

OBJS_SRC =  \
BatchLinearAlgebra.o \

OBJS = $(OBJS_SRC)

all : $(LIB_BATCHED_BLAS_SO)

$(LIB_BATCHED_BLAS_SO) : $(OBJS)
	$(CC) -fPIC -shared -o $(LIB_BATCHED_BLAS_SO) $(OBJS)

.cpp.o :
	$(CC) $(CCFLAG) -c $< $(DEF)

clean :
	rm -f $(LIB_BATCHED_BLAS_SO) *.o