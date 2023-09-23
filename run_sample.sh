# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run sample script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

rm -rf log ncu && mkdir -p log ncu

# $1: M. $2: L, $3: N, $4: K
evaluate_b2b_hgemm() {
    echo "Evaluating $1 * $2 * $3 * $4"
    $WORK_PATH/output/bin/b2b_hgemm -M=$1 -L=$2 -N=$3 -K=$4 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/b2b_hgemm_${1}_${2}_${3}_${4}.log 2>&1
    sleep 3
}

# $1: M. $2: L, $3: N, $4: K
ncu_b2b_hgemm() {
    echo "NCU $1 * $2 * $3 * $4"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/b2b_hgemm_${1}_${2}_${3}_${4} $WORK_PATH/output/bin/b2b_hgemm -M=$1 -L=$2 -N=$3 -K=$4 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_b2b_hgemm_${1}_${2}_${3}_${4}.log 2>&1
    sleep 3
}

benchmark_b2b_hgemm() {
    M_dims=(128 256 512 768 1024 1536 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 13312 14336 15360 16384)
    L=8
    N=16
    K=512

    for M in ${M_dims[@]};
    do
        evaluate_b2b_hgemm $M $L $N $K
        # ncu_b2b_hgemm $M $L $N $K
    done
}

nohup $WORK_PATH/output/bin/b2b_hgemm -M=512 -L=2048 -N=16 -K=1024 -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/b2b_hgemm_512_2048_16_1024.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/b2b_hgemm_512_2048_16_1024 $WORK_PATH/output/bin/b2b_hgemm -M=512 -L=2048 -N=16 -K=1024 -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_b2b_hgemm_512_2048_16_1024.log 2>&1

# benchmark_b2b_hgemm
