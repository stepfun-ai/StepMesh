export BINARY=${BINARY:-./cmake_build/tests/test_benchmark_af_tensor}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

set -x

function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    pkill -9 -f $BINARY
    sleep 1
}
trap cleanup EXIT
cleanup # cleanup on startup

export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-1}
export DMLC_NUM_SERVER=1
export DMLC_NODE_RANK=${DMLC_NODE_RANK:-0}
export DMLC_SPLIT_QP_LAG=0
export DMLC_INTERFACE=brainpf_bond0        # my RDMA interface
export DMLC_PS_ROOT_URI=$(ip -o -4 addr | grep ${DMLC_INTERFACE} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-12278} # scheduler's port (can random choose)
export DMLC_ENABLE_RDMA=ibverbs
export PS_VERBOSE=${PS_VERBOSE:-1}

export BYTEPS_ENABLE_IPC=0
export BENCHMARK_NTHREAD=${BENCHMARK_NTHREAD:-1}

if [ $# -eq 0 ]
then
    echo "SCHEDULER_IP is ${DMLC_PS_ROOT_URI}"
    # launch scheduler
    export DMLC_NODE_HOST=${DMLC_PS_ROOT_URI}
    DMLC_ROLE=scheduler $BINARY &
    export BENCHMARK_GPU=0
    DMLC_ROLE=server gdb --ex "run" --args $BINARY
else
    export DMLC_INTERFACE=auto
    export BENCHMARK_WORKER_NUMBER=1
    export BENCHMARK_WORKER_RANK=0
    export BENCHMARK_GPU=0
    export DMLC_PS_ROOT_URI=${1}
    echo "Using ${1} as DMLC_PS_ROOT_URI"
    DMLC_ROLE=worker  $BINARY
fi

