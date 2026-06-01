export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON. 
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

echo "===== $1, $2, $3 ====="
# the number of gpus is the second input argument, should be equal to nproc_per_node
torchrun --rdzv-id=101  --rdzv-backend=c10d  --rdzv-endpoint=$3:1234 \
         --nnodes=$1 --nproc_per_node=$2 \
        ${megatron_path}/pretrain_gpt.py \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50304 \
        \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --data-parallel-sharding-strategy no_shard \
        \
        --micro-batch-size 8 \
        --global-batch-size 128 \
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


