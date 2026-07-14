export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache

# B200 single-node variant of run-1node.sh. Same model/hyperparameters, but the
# global batch size is computed from the GPU count so a 1..8 GPU scan is valid
# for every count (Megatron requires global_batch % (micro_batch * dp) == 0;
# the hardcoded 128 in run-1node.sh fails for 3, 5, 6, 7 GPUs).
#
# the number of gpus is the first input argument, equal to nproc_per_node = dp
NPROC=$1
MICRO=8
GRAD_ACC=16                                  # grad-accum steps (fixed => weak scaling)
GLOBAL=$(( MICRO * NPROC * GRAD_ACC ))       # per-GPU work fixed; scales with #GPUs

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
