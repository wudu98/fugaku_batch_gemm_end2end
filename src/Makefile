CC=fcc
CCFLAG=-std=gnu++11 -Nclang -Kfast,ocl,openmp -fPIC
BLAS=-SSL2BLAMP -lm -lfjomp

LIB_BATCHED_BLAS_SO = libsgemm_intercept.so

OBJS_SRC =  \
sgemm_intercept.o \

OBJS = $(OBJS_SRC)

all : $(LIB_BATCHED_BLAS_SO)

$(LIB_BATCHED_BLAS_SO) : $(OBJS)
	$(CC) -fPIC -shared -o $(LIB_BATCHED_BLAS_SO) $(OBJS)

.cpp.o :
	$(CC) $(CCFLAG) -c $< $(DEF)

clean :
	rm -f $(LIB_BATCHED_BLAS_SO) *.o