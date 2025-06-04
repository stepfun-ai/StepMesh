export role=$1
export NIC=$2
echo $role
echo $NIC
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_NUM_WORKER=1
export DMLC_SPLIT_QP_LAG=1
export DMLC_NUM_SERVER=1 
# export TOTAL_DURATION=10000
# export PS_VERBOSE=2
# export NUM_KEY_PER_SERVER=1
export PS_QP_NUM=2
export DMLC_PS_ROOT_URI=`ip addr show brainpf0 | awk '/inet / {print $2}' | cut -d/ -f1`
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_INTERFACE=$NIC        # my RDMA interface 
export DMLC_NODE_HOST=`ip addr show $DMLC_INTERFACE | awk '/inet / {print $2}' | cut -d/ -f1`
export DMLC_RANK=$3
if [ "$role" = "server" ]; then
    echo "launching server"
    DMLC_ROLE=server ./tests/test_benchmark 1024000 10 1
elif [ "$role" = "worker" ]; then
# export DMLC_SPLIT_QP_LAG=$4
    echo "launching worker"
    DMLC_ROLE=worker ./tests/test_benchmark 1024000 10 1
elif [ "$role" = "scheduler" ]; then
    echo "launching scheduler"
    DMLC_ROLE=scheduler  ./tests/test_benchmark 1024000 10 1
else
    echo "Unknown role: $role"
    exit 1
fi