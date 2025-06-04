export BINARY=${BINARY:-./tests/test_benchmark_af_gdr}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

set -x

function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    pkill -9 -f $BINARY
    sleep 1
}
trap cleanup EXIT
# cleanup # cleanup on startup

export DMLC_NUM_WORKER=1 #${DMLC_NUM_WORKER:-1}
export DMLC_NUM_SERVER=1
export DMLC_SPLIT_QP_LAG=1
export DMLC_INTERFACE=brainpf_bond2
export SCHEDULER_IP=`ip addr show $DMLC_INTERFACE  | awk '/inet / {print $2}' | cut -d/ -f1`
export DMLC_PS_ROOT_URI=${SCHEDULER_IP}

export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-12278} # scheduler's port (can random choose)

export DMLC_ENABLE_RDMA=ibverbs
export PS_VERBOSE=3
export BYTEPS_ENABLE_IPC=0
export BENCHMARK_NTHREAD=${BENCHMARK_NTHREAD:-1}

if [ $# -eq 0 ]
then
    # launch scheduler
    export DMLC_NODE_HOST=${SCHEDULER_IP}
    DMLC_ROLE=scheduler  $BINARY &
    export BENCHMARK_GPU=0
    # DMLC_ROLE=server $BINARY
    DMLC_ROLE=server gdb --ex "run"  --args  $BINARY
else
    export DMLC_INTERFACE=brainpf_bond3
    export DMLC_NODE_HOST=`ip addr show $DMLC_INTERFACE | awk '/inet / {print $2}' | cut -d/ -f1`
    export BENCHMARK_WORKER_NUMBER=1
    export BENCHMARK_WORKER_RANK=0
    export BENCHMARK_GPU=0
    # DMLC_ROLE=worker $BINARY

    DMLC_ROLE=worker  $BINARY
fi

