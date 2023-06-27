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

PS=128
HS=7168
SL=2048
NH=56
TB=1
MP=(1 2 4 8 16 32)

source /vol0004/ra000012/data/wahib/gnn/wudu/LLM_speedup/benchmark/dl4fugaku/build/venv/bin/activate

# python ./test/test.py
LD_PRELOAD=$(realpath ./src/libsgemm_intercept.so) python ./test/test.py
