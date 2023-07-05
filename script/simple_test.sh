#!/bin/bash
set -e

module switch lang/tcsds-1.2.37

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/..; pwd`
cd ${PROJECT_ROOT}

threads=$1
export OMP_NUM_THREADS=${threads}

MPIEXEC=""
if [ $threads -eq 48 ]; then
	MPIEXEC="mpiexec -mca plm_ple_memory_allocation_policy interleave_all"
fi

cd ./benchmark
make clean
make -s
cd ..

layout_=1
transa_=0
transb_=0
parallel_mode_=0

echo  "layout_: " $layout_ $transa_ $transb_ $parallel_mode_ 
TB=1
B=12
M=2048
N=2048
K=128
echo -n $TB "," $B "," $M "," $N "," $K ","
$MPIEXEC ./benchmark/batch_gemm_benchmark $TB $B $M $N $K $layout_ $transa_ $transb_ $parallel_mode_ 
