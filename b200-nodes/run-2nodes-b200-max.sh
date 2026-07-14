export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache

# High-utilization two-node variant of run-1node-b200-max.sh (same ~5B model,
# full activation recompute, large GEMMs). Inter-node gradient all-reduce runs
# over the 8 B200 NDR (400 Gb/s) rails.

# Force NCCL onto the B200 NDR InfiniBand rails (not the HDR100 / down NICs)
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_DEBUG=INFO

# Args: $1 = nnodes, $2 = nproc_per_node, $3 = master ip, $4 = precision (bf16|fp8)
NNODES=$1
NPROC=$2
MASTER=$3
PREC="${4:-bf16}"
MICRO=8
GRAD_ACC=8
DP=$(( NNODES * NPROC ))                 # tp=pp=1, so dp = world size
GLOBAL=$(( MICRO * DP * GRAD_ACC ))

if [ "$PREC" = "fp8" ]; then
   PREC_ARGS="--bf16 --transformer-impl transformer_engine \
      --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
else
   PREC_ARGS="--bf16"
fi

echo "===== nnodes=$NNODES nproc=$NPROC master=$MASTER precision=$PREC dp=$DP global=$GLOBAL ====="
torchrun --rdzv-id=201  --rdzv-backend=c10d  --rdzv-endpoint=$MASTER:1234 \
         --nnodes=$NNODES --nproc_per_node=$NPROC \
        ${megatron_path}/pretrain_gpt.py \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50304 \
        \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --use-distributed-optimizer \
        \
        --micro-batch-size $MICRO \
        --global-batch-size $GLOBAL \
        \
        --num-layers 24 \
        --hidden-size 4096 \
        --num-attention-heads 32 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        \
        --recompute-granularity full \
        --recompute-method uniform \
        --recompute-num-layers 1 \
        \
        --train-iters 50 \
        --lr 3e-4 \
        --min-lr 3e-5 \
        --lr-decay-style cosine \
        --lr-warmup-iters 3 \
        --lr-decay-iters 45 \
        \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        \
        $PREC_ARGS \
        \
        --eval-interval 1000000 \
        --save-interval 1000000 \
        --log-interval 10 \
	--log-throughput \
	--timing-log-level 2
