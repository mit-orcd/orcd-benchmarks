export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache

# High-utilization single-node variant: a ~5B GPT sized to keep the B200 tensor
# cores busy (large GEMMs) while staying well inside 180 GB HBM.
#   - N=1 is the worst case for memory (pure data parallel, each GPU holds the
#     full model + optimizer). ~5B params => ~90 GB weights+Adam; full activation
#     recomputation keeps activations small => ~120 GB total, ~60 GB margin.
#   - Large tokens/microstep (micro_batch * seq = 8 * 4096) => big matmuls => high
#     achieved TFLOP/s. Read the per-GPU number on the "--log-throughput" lines.
#
# Args:  $1 = nproc_per_node (= dp),  $2 = precision (bf16 | fp8, default bf16)
NPROC=$1
PREC="${2:-bf16}"
MICRO=8
GRAD_ACC=8
GLOBAL=$(( MICRO * NPROC * GRAD_ACC ))

# fp8 (Transformer Engine) roughly doubles tensor-core throughput on B200; bf16
# is the robust default. fp8 keeps a bf16 base with fp8 GEMMs.
if [ "$PREC" = "fp8" ]; then
   PREC_ARGS="--bf16 --transformer-impl transformer_engine \
      --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
else
   PREC_ARGS="--bf16"
fi

echo "===== nproc=$NPROC precision=$PREC micro=$MICRO global=$GLOBAL ====="
torchrun --standalone --nproc_per_node=$NPROC \
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
