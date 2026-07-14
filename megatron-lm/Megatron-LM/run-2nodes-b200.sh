export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache

# B200 two-node variant of run-2nodes.sh. Two differences from the original:
#   1. NCCL_IB_HCA is set to the 8 B200 NDR (400 Gb/s) GPU rails. The original
#      lists mlx5_0..mlx5_8, which on these nodes are HDR100 / down NICs.
#   2. global batch size is computed from the world size so a 1..8 GPU/node scan
#      is valid for every count (see run-1node-b200.sh for the reason).

# Force NCCL to use InfiniBand for inter-node communication (B200 NDR rails only)
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=INFO

# args: $1 = nnodes, $2 = nproc_per_node (gpus/node), $3 = master node ip
NNODES=$1
NPROC=$2
MASTER=$3
MICRO=8
GRAD_ACC=16                                       # fixed => weak scaling
DP=$(( NNODES * NPROC ))                           # tp=pp=1, so dp = world size
GLOBAL=$(( MICRO * DP * GRAD_ACC ))

echo "===== nnodes=$NNODES nproc_per_node=$NPROC master=$MASTER dp=$DP global_batch=$GLOBAL ====="
torchrun --rdzv-id=101  --rdzv-backend=c10d  --rdzv-endpoint=$MASTER:1234 \
         --nnodes=$NNODES --nproc_per_node=$NPROC \
        ${megatron_path}/pretrain_gpt.py \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50304 \
        \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --data-parallel-sharding-strategy no_shard \
        \
        --micro-batch-size $MICRO \
        --global-batch-size $GLOBAL \
        \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        \
        --train-iters 2000 \
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
        --log-interval 20 \
	--log-throughput \
	--timing-log-level 2
