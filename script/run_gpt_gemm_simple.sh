#!/bin/bash
set -e

module switch lang/tcsds-1.2.37

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

threads=$1
export OMP_NUM_THREADS=${threads}

MPIEXEC=""
if [ $threads -eq 48 ]; then
	MPIEXEC="mpiexec -mca plm_ple_memory_allocation_policy interleave_all"
fi

PS=128
HS=7168
SL=2048
NH=56
TB=1
MP=(1 2 4 8 16 32)

source /vol0004/ra000012/data/wahib/gnn/wudu/LLM_speedup/benchmark/dl4fugaku/build/venv/bin/activate

python ./test/test.py
LD_PRELOAD=$(realpath ./src/libsgemm_intercept.so) python ./test/test.py
# cd ./benchmark
# make clean
# make -s
# cd ..

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./benchmark

# layout_=1
# transa_=0
# transb_=0
# parallel_mode_=0

# echo  "layout_: " $layout_ $transa_ $transb_ $parallel_mode_ 
# for (( i=0; i<6; i++))
# do
#     B=1
#     M=${SL}
#     N=$(( ${PS} * ${NH} * 3 / ${MP[$i]} ))
#     K=${HS}
#     echo -n $TB "," $B "," $M "," $N "," $K ","
#     $MPIEXEC ./benchmark/batch_gemm_benchmark $TB $B $M $N $K $layout_ $transa_ $transb_ 0
# done

# for (( i=0; i<4; i++))
# do
#     B=$(( ${NH} / ${MP[$i]} ))
#     M=${SL}
#     N=${SL}
#     K=${PS}
#     echo -n $TB "," $B "," $M "," $N "," $K ","
#     $MPIEXEC ./benchmark/batch_gemm_benchmark $TB $B $M $N $K $layout_ $transa_ $transb_ 0
# done

# for (( i=0; i<4; i++))
# do
#     B=$(( ${NH} / ${MP[$i]} ))
#     M=${SL}
#     N=${PS}
#     K=${SL}
#     echo -n $TB "," $B "," $M "," $N "," $K ","
#     $MPIEXEC ./benchmark/batch_gemm_benchmark $TB $B $M $N $K $layout_ $transa_ $transb_ 0
# done

# for (( i=0; i<6; i++))
# do
#     B=1
#     M=${SL}
#     N=${HS}
#     K=$(( ${HS} / ${MP[$i]} ))
#     echo -n $TB "," $B "," $M "," $N "," $K ","
#     $MPIEXEC ./benchmark/batch_gemm_benchmark $TB $B $M $N $K $layout_ $transa_ $transb_ 0
# done

# for (( i=0; i<6; i++))
# do
#     B=1
#     M=${SL}
#     N=$(( 4 * ${HS} / ${MP[$i]} ))
#     K=${HS}
#     echo -n $TB "," $B "," $M "," $N "," $K ","
#     $MPIEXEC ./benchmark/batch_gemm_benchmark $TB $B $M $N $K $layout_ $transa_ $transb_ 0
# done

# for (( i=0; i<6; i++))
# do
#     B=1
#     M=${SL}
#     N=${HS}
#     K=$(( 4 * ${HS} / ${MP[$i]} ))
#     echo -n $TB "," $B "," $M "," $N "," $K ","
#     $MPIEXEC ./benchmark/batch_gemm_benchmark $TB $B $M $N $K $layout_ $transa_ $transb_ 0
# done
