export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache

# Apples-to-apple with ~/data022/aicr-benchmarks/Benchmark_WG/megatron-lm (B200):
# same ~7B GPT model, micro-batch 4, global batch = 128 x total_GPUs, bf16,
# 100 iters, NO activation recompute, no_shard optimizer. Only difference from
# the reference is the cluster (nodes/partition). Read the per-GPU TFLOP/s on the
# "--log-throughput" lines.
#
# the number of gpus is the first input argument, equal to nproc_per_node = dp
NPROC=$1
MICRO=4
GLOBAL=$(( 128 * NPROC ))          # reference formula: 128 x total GPUs

# reference B200 model (~7B params)
model_par="--num-layers 36 \
   --hidden-size 4096 \
   --ffn-hidden-size 14336 \
   --num-attention-heads 32 \
   --seq-length 2048 \
   --max-position-embeddings 2048"

echo "===== nproc=$NPROC micro=$MICRO global=$GLOBAL ====="
torchrun --standalone --nproc_per_node=$NPROC \
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
