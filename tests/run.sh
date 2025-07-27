export BINARY=${BINARY:-./cmake_build/tests/stepmesh_echo_test}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    pkill -9 -f $BINARY
    sleep 1
}
trap cleanup EXIT
cleanup

export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export STEPMESH_SPLIT_QP_LAG=1
export DMLC_INTERFACE=${RNIC:-brainpf_bond0}
export SCHEDULER_IP=`ip addr show $DMLC_INTERFACE  | awk '/inet / {print $2}' | cut -d/ -f1`
export DMLC_PS_ROOT_URI=${SCHEDULER_IP}
export DMLC_PS_ROOT_PORT=${DMLC_PS_ROOT_PORT:-12278}
export DMLC_ENABLE_RDMA=ibverbs
#export PS_VERBOSE=3
ROLE=${ROLE:-server}
if [ $ROLE == "server" ]; then
  echo "Run server and scheduler, scheduler ip $SCHEDULER_IP "
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  DMLC_ROLE=scheduler $BINARY &
  export DMLC_INTERFACE=auto
  DMLC_ROLE=server STEPMESH_GPU=0 $BINARY
elif [ $ROLE == "worker" ]; then
  echo "Run worker with scheduler ip: $1"
  export DMLC_PS_ROOT_URI=$1
  export DMLC_INTERFACE=auto
  DMLC_ROLE=worker STEPMESH_GPU=0 $BINARY
elif [ $ROLE == "joint" ]; then
  echo "Run scheduler, server, and worker jointly"
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  export DMLC_PS_ROOT_URI=$SCHEDULER_IP
  DMLC_ROLE=scheduler $BINARY &
  export DMLC_INTERFACE=auto
  DMLC_ROLE=server STEPMESH_GPU=0 $BINARY &
  sleep 10
  export DMLC_INTERFACE=auto
  DMLC_ROLE=worker STEPMESH_GPU=0  $BINARY
fi
