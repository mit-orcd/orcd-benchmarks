export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

#Optional but often useful in containers. Set path to cuda driver libs for compiling pytorchInductor and TRITON.
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
export LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/torch_extensions
export XDG_CACHE_HOME=$PWD/xdg_cache
# per-job temp/cache dirs under the user scratch dir (set by the job script)
if [ -n "$SCR" ]; then
   export TMPDIR=$SCR/tmp TORCHINDUCTOR_CACHE_DIR=$SCR/inductor TRITON_CACHE_DIR=$SCR/triton
   export XDG_CACHE_HOME=$SCR/xdg_cache
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Tuned "max throughput" single-node Megatron point, one per job. The sweep
# (job-megatron-max-sweep.sh) runs a grid of these on B200 and B300 and the
# best TFLOP/s/GPU per GPU type is compared -- the two GPU types do NOT need to
# share a config. The knobs trade memory for FLOPs:
#   - recompute none uses the most memory and wastes no FLOPs; full recompute
#     re-runs each layer's forward (~25-30% extra compute, not counted in the
#     reported model TFLOP/s) but frees memory for bigger micro-batches.
#   - bigger micro-batch / bigger model => bigger GEMMs => higher utilization.
#   - distributed optimizer + overlapped grad-reduce / param-gather keep the
#     static memory and the communication cost small on 8 GPUs.
# OOM points are expected at the top of the grid; they show where memory ends.
#
# Args: $1 nproc  $2 model (5b|13b)  $3 micro-batch  $4 recompute (none|selective|full)
#       $5 precision (bf16|fp8)
NPROC=$1
MODEL=$2
MICRO=$3
RECOMP=$4
PREC=$5
GRAD_ACC=4
GLOBAL=$(( MICRO * NPROC * GRAD_ACC ))

case "$MODEL" in
   5b)  MODEL_ARGS="--num-layers 24 --hidden-size 4096 --ffn-hidden-size 16384 --num-attention-heads 32" ;;
   13b) MODEL_ARGS="--num-layers 40 --hidden-size 5120 --ffn-hidden-size 20480 --num-attention-heads 40" ;;
   *) echo "unknown model $MODEL"; exit 1 ;;
esac

case "$RECOMP" in
   none)      RECOMP_ARGS="" ;;
   selective) RECOMP_ARGS="--recompute-granularity selective" ;;
   full)      RECOMP_ARGS="--recompute-granularity full --recompute-method uniform --recompute-num-layers 1" ;;
   *) echo "unknown recompute $RECOMP"; exit 1 ;;
esac

if [ "$PREC" = "fp8" ]; then
   PREC_ARGS="--bf16 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
else
   PREC_ARGS="--bf16"
fi

echo "===== nproc=$NPROC model=$MODEL micro=$MICRO recompute=$RECOMP precision=$PREC global=$GLOBAL ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
torchrun --standalone --nproc_per_node=$NPROC \
        ${megatron_path}/pretrain_gpt.py \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50304 \
        \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --transformer-impl transformer_engine \
        \
        --micro-batch-size $MICRO \
        --global-batch-size $GLOBAL \
        \
        $MODEL_ARGS \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        \
        $RECOMP_ARGS \
        \
        --train-iters 20 \
        --lr 3e-4 \
        --min-lr 3e-5 \
        --lr-decay-style cosine \
        --lr-warmup-iters 2 \
        --lr-decay-iters 18 \
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
        --log-interval 5 \
	--log-throughput \
	--timing-log-level 0
echo "exit code: $?"
