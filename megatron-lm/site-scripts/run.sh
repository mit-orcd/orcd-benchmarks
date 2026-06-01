export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

# Set path to cuda driver libs for compiling PyTorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache

# Force NCCL to use InfiniBand for inter-node communication
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=INFO

# Arguments:
#   $1 = N_NODES
#   $2 = N_GPUS (per node)
#   $3 = master_node_ip
#   $4 = GPU_TYPE  (h200 or l40s)
#   $5 = GLOBAL_BATCH_SIZE

N_NODES=$1
N_GPUS=$2
MASTER_IP=$3
GPU_TYPE=$4
GBS=${5:?ERROR: global batch size must be passed as fifth argument to run.sh}

# micro-batch-size stays at 4; gradient accumulation = GBS / (4 x DP) = GBS / (4 x N_NODES x N_GPUS)
MICRO_BS=4

# Model size by GPU type
if [ "$GPU_TYPE" = "l40s" ]; then
    model_par="--num-layers 24 \
   --hidden-size 2048 \
   --ffn-hidden-size 8192 \
   --num-attention-heads 16 \
   --seq-length 2048 \
   --max-position-embeddings 2048"
elif [ "$GPU_TYPE" = "h200" ]; then
    model_par="--num-layers 24 \
   --hidden-size 4096 \
   --ffn-hidden-size 16384 \
   --num-attention-heads 32 \
   --seq-length 2048 \
   --max-position-embeddings 2048"
else
    echo "====== GPU type error: unknown GPU_TYPE='$GPU_TYPE' ======"
    exit 1
fi
echo "Model parameters"
echo "$model_par"

# Network rendezvous: standalone for single node, c10d for multi-node
if [ "$N_NODES" = "1" ]; then
    rdzv_info="--standalone"
else
    rdzv_info="--rdzv-id=101  --rdzv-backend=c10d  --rdzv-endpoint=${MASTER_IP}:1234"
fi
echo "Network info: $rdzv_info"
echo "Global batch size: $GBS  |  Micro batch size: $MICRO_BS"

torchrun $rdzv_info \
        --nnodes=$N_NODES --nproc_per_node=$N_GPUS \
        ${megatron_path}/pretrain_gpt.py \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50304 \
        \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --data-parallel-sharding-strategy no_shard \
        \
        --micro-batch-size $MICRO_BS \
        --global-batch-size $GBS \
        \
        $model_par \
        \
        --train-iters 100 \
        --lr 3e-4 \
        --min-lr 3e-5 \
        --lr-decay-style cosine \
        --lr-warmup-iters 10 \
        --lr-decay-iters 50 \
        \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        \
        --bf16 \
        \
        --eval-interval 1000000 \
        --save-interval 1000000 \
        --log-interval 10 \
        --log-throughput \
        --timing-log-level 2 \
        --timing-log-option all
