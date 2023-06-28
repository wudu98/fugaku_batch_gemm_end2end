#!/bin/bash
set -e

module switch lang/tcsds-1.2.37

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/..; pwd`
cd ${PROJECT_ROOT}

threads=$1
export OMP_NUM_THREADS=${threads}

source /vol0004/ra000012/data/wahib/gnn/wudu/LLM_speedup/benchmark/dl4fugaku/build/venv/bin/activate

python ./test/test.py
# LD_PRELOAD=$(realpath ./src/libbatch_sgemm_intercept.so) python ./test/test.py
