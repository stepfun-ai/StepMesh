THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    # pkill -9 -f test_bench
    pkill -9 -f python3
    sleep 1
}
trap cleanup EXIT
cleanup

echo $1

export BIN=${BIN:-test_remote_moe}
# common setup
export DMLC_INTERFACE=brainpf_bond0
export SCHEDULER_IP=$(ip -o -4 addr | grep ${DMLC_INTERFACE} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_NUM_WORKER=${NUM_WORKER:-1}

export DMLC_NUM_SERVER=${NUM_SERVER:-1}

export DMLC_GROUP_SIZE=8

export DMLC_NODE_RANK=${NODE_RANK:-0}
export DMLC_PS_ROOT_PORT=8123
export DMLC_PS_ROOT_URI=$SCHEDULER_IP  # scheduler's RDMA interface IP
export DMLC_ENABLE_RDMA=ibverbs
export NCCL_DEBUG=warning
export DMLC_SPLIT_QP_LAG=1

export PS_VERBOSE=0

r=${r:-server}
if [ $r == "server" ]; then
  if [ $NODE_RANK == 0 ]; then
    echo "Scheduler ip $SCHEDULER_IP "
    export DMLC_NODE_HOST=${SCHEDULER_IP}
    DMLC_ROLE=scheduler python3 $THIS_DIR/${BIN}.py &
    export DMLC_PS_ROOT_URI=$SCHEDULER_IP
  else
    export DMLC_PS_ROOT_URI=$1
    echo "Using scheduler ip: $1"
  fi
  export DMLC_INTERFACE=auto
  for P in {0..7}; do
    DMLC_ROLE=server STEPAF_GPU=${P} numactl --membind=netdev:brainpf${P}_0 --cpubind=netdev:brainpf${P}_0  python3 $THIS_DIR/${BIN}.py "${@:2}" &
  done
elif [ $r == "worker" ]; then
  echo "Using scheduler ip: $1"
  export DMLC_PS_ROOT_URI=$1
  export DMLC_INTERFACE=auto
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  for P in {0..7}; do
    DMLC_ROLE=worker STEPAF_GPU=${P} numactl --membind=netdev:brainpf${P}_0 --cpubind=netdev:brainpf${P}_0  python3 $THIS_DIR/${BIN}.py "${@:2}" &
  done
elif [ $r == "local" ]; then
  export DMLC_GROUP_SIZE=1
  echo "local test scheduler ip $SCHEDULER_IP "
  export DMLC_NODE_HOST=${SCHEDULER_IP}
  export DMLC_PS_ROOT_URI=$SCHEDULER_IP

  if [ $1 == "sc" ]; then
    echo "Starting scheduler"
    DMLC_ROLE=scheduler python3 $THIS_DIR/${BIN}.py &
  fi

  echo "Using scheduler ip: $2"
  # export DMLC_PS_ROOT_URI=$2
  export DMLC_INTERFACE=auto
  export DMLC_NODE_HOST=${SCHEDULER_IP}

  if [ $1 == "se0" ]; then
  
  DMLC_ROLE=server STEPAF_GPU=0 python3 $THIS_DIR/${BIN}.py &
  fi

  if [ $1 == "se1" ]; then
  export DMLC_INSTANCE_ID=0
 
  DMLC_ROLE=server STEPAF_GPU=1  python3 $THIS_DIR/${BIN}.py &
  fi

  if [ $1 == "w" ]; then
   DMLC_ROLE=worker STEPAF_GPU=0 python3 $THIS_DIR/${BIN}.py &
  fi
fi

wait