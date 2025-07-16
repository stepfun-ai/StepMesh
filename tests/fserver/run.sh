THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    # pkill -9 -f test_bench
    pkill -9 -f test_remote_moe
    pkill -9 -f test_fserver
    sleep 1
}
trap cleanup EXIT
cleanup

# common setup
export BIN=${BIN:-test_fserver}
export SCHEDULER_IP=$(ip -o -4 addr | grep brainpf_bond0 | awk '{print $4}' | cut -d'/' -f1) 
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=$SCHEDULER_IP  # scheduler's RDMA interface IP 
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_INTERFACE=auto

export DMLC_NODE_HOST=${SCHEDULER_IP}
export DMLC_INTERFACE=auto
export DMLC_SPLIT_QP_LAG=1
export STEPPS_GPU=0
# export PS_VERBOSE=2

# if [ $r == "server" ]; then
#     DMLC_ROLE=scheduler python3 $THIS_DIR/test_fserver.py & 
#     DMLC_ROLE=server python3 $THIS_DIR/test_fserver.py 
#     # DMLC_ROLE=server gdb -ex 'run' -ex 'bt' -batch --args  python3 $THIS_DIR/test_fserver.py & 
# else
#     # DMLC_ROLE=worker python3 $THIS_DIR/test_fserver.py
#     DMLC_ROLE=worker ./ps-lite/cmake_build/tests/utests/ut_tensor_worker
#     # DMLC_ROLE=worker $THIS_DIR/../ps-lite/tests/test_benchmark_af
# fi

DMLC_ROLE=scheduler python3 $THIS_DIR/$BIN.py &
DMLC_ROLE=server python3 $THIS_DIR/$BIN.py $@ &
DMLC_ROLE=worker python3 $THIS_DIR/$BIN.py $@

wait