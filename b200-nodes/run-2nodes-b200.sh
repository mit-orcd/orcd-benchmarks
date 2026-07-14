export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache

# Apples-to-apple with ~/data022/aicr-benchmarks/Benchmark_WG/megatron-lm (B200),
# two-node case: same ~7B GPT model, micro-batch 4, global batch = 128 x total_GPUs
# (= 128 x 2 x gpus_per_node), bf16, 100 iters, NO activation recompute, no_shard.
# NCCL_IB_HCA is set to the 8 B200 NDR (400 Gb/s) rails (the reference's
# mlx5_0..mlx5_8 are HDR100 / down NICs on these nodes) — an infrastructure fix
# for this cluster, not a change to the benchmark definition.

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
MICRO=4
DP=$(( NNODES * NPROC ))            # tp=pp=1, so dp = total GPUs
GLOBAL=$(( 128 * DP ))             # reference formula: 128 x total GPUs

# reference B200 model (~7B params)
model_par="--num-layers 36 \
   --hidden-size 4096 \
   --ffn-hidden-size 14336 \
   --num-attention-heads 32 \
   --seq-length 2048 \
   --max-position-embeddings 2048"

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
