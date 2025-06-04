export SCHEDULER_BIN=${SCHEDULER_BIN:-./cmake_build/tests/utests/ut_scheduler}
export SERVER_BIN=${SERVER_BIN:-./cmake_build/tests/utests/ut_server}
export WORKER_BIN=${WORKER_BIN:-./cmake_build/tests/utests/ut_tensor_worker}

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cmd=$1
echo $cmd
set -x

function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    pkill -9 -f $SERVER_BIN
    pkill -9 -f $SCHEDULER_BIN
    pkill -9 -f $WORKER_BIN
    sleep 1
}
trap cleanup EXIT

export DMLC_NUM_WORKER=${DMLC_NUM_WORKER:-1}
export DMLC_NUM_SERVER=1
export DMLC_INTERFACE=brainpf_bond0        # my RDMA interface
export DMLC_PS_ROOT_URI=$(ip -o -4 addr | grep ${DMLC_INTERFACE} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-12278} # scheduler's port (can random choose)
export DMLC_SPLIT_QP_LAG=0
# export PS_VERBOSE=3
export DMLC_ENABLE_RDMA=ibverbs


echo "SCHEDULER_IP is ${DMLC_PS_ROOT_URI}"

# # launch scheduler
export DMLC_NODE_HOST=${DMLC_PS_ROOT_URI}
# if [ "$cmd" == "sc" ]; then
cleanup # cleanup on startup
DMLC_ROLE=scheduler $SCHEDULER_BIN &
# fi


# if [ "$cmd" == "se0" ]; then
export STEPAF_GPU=0
# DMLC_ROLE=server gdb --ex "run" --args $SERVER_BIN
DMLC_ROLE=server $SERVER_BIN &
# fi 


export STEPAF_GPU=1
export DMLC_INTERFACE=auto

# if [ "$cmd" == "w" ]; then
# DMLC_ROLE=worker gdb --ex "run" --args  $WORKER_BIN
DMLC_ROLE=worker $WORKER_BIN

# fi
wait
