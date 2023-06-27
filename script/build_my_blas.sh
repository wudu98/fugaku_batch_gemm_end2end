#!/bin/bash
set -e

module switch lang/tcsds-1.2.37

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

cd ./src
make clean
make 
