THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    # pkill -9 -f test_bench
    pkill -9 -f test_remote_moe
    pkill -9 -f test_fserver
    sleep 1
}
trap cleanup EXIT
# cleanup

# common setup
export BIN=${BIN:-test_fserver}
# export DMLC_INTERFACE=${RNIC:-brainpf_bond0}
export SCHEDULER_IP=$(ip -o -4 addr | grep ${RNIC} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=$SCHEDULER_IP  # scheduler's RDMA interface IP 
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_INTERFACE=auto
# export STEPMESH_BIND_CPU_CORE=1

export DMLC_NODE_HOST=${SCHEDULER_IP}
export DMLC_INTERFACE=auto
export STEPMESH_SPLIT_QP_LAG=0
export STEPMESH_BIND_CPU_CORE=1
export STEPMESH_GPU=0
export PS_VERBOSE=1

DMLC_ROLE=scheduler numactl -m 0 python3 $THIS_DIR/$BIN.py &
export STEPMESH_CPU_START_OFFSET=10
DMLC_ROLE=server numactl -m 0 python3 $THIS_DIR/$BIN.py $@ &
# DMLC_ROLE=worker python3 $THIS_DIR/$BIN.py $@ &
# export STEPMESH_DROP_RATE=1
export STEPMESH_CPU_START_OFFSET=15
DMLC_ROLE=worker numactl -m 0 python3 $THIS_DIR/$BIN.py $@

wait
