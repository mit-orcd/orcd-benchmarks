export work_path="/orcd/data/orcd/022/benchmarks/megatron-lm"
export megatron_path="$work_path/Megatron-LM"

torchrun --standalone --nproc_per_node=2 \
        ${megatron_path}/pretrain_gpt.py \
        --mock-data \
        --tokenizer-type NullTokenizer \
        --vocab-size 50304 \
        \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --data-parallel-sharding-strategy no_shard \
        \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        \
        --micro-batch-size 4 \
        --global-batch-size 8 \
        \
        --train-iters 50 \
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
        --log-interval 1 \
        --eval-interval 1000000 \
        --save-interval 1000000 \
	--timing-log-level 2 \
        --no-save-optim --no-save-rng --no-load-optim --no-load-rng

#	--log-throughput \

# --standalone avoids needing MASTER_ADDR/MASTER_PORT.

# With 2 GPUs and data-parallel-size=2, ensure global-batch-size is divisible by micro-batch-size * 2 (e.g., 8 divisible by 4*2).
# If your GPUs don’t support BF16, replace --bf16 with --fp16. Other options: fp32, fp8.

# Setting both to 1 forces all GPUs to act as Data Parallel replicas
#DIST_ARGS="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1"

# disable the Inductor/torch.compile path
#export TORCHINDUCTOR_DISABLE=1

# Using --mock-data to bypass physical dataset requirements
# use no takenizer for benchmarks
#        --tokenizer-type NullTokenizer \
#        --vocab-size 50304 \

# set rdzv
# --rdzv_id=$SLURM_JOB_ID --rdzv_endpoint="localhost:1230" \

# do not bind home dir, because it has the conda env that conflict with container
#    --bind /home/shaohao \

# internal tokenizer
#    --tokenizer-type GPT2BPETokenizer \
#    --vocab-file ${megatron_path}/data/vocab.json \
#    --merge-file ${megatron_path}/data/merges.txt

# do not use this, because it will use HuggingFace transformers library no matter what
# --tokenizer-model-type gpt2 

# standard data parallel
#--data-parallel-sharding-strategy no_shard

# Megatron FSDP (~15% faster than PyTorch FSDP2)
# --use-megatron-fsdp \
# --data-parallel-sharding-strategy optim_grads_params

# TP
#--tensor-model-parallel-size 4  # 4-way tensor parallelism
#--sequence-parallel              # Enable sequence parallelism (recommended)
# PP
#--pipeline-model-parallel-size 8              # 8 pipeline stages
#--num-layers-per-virtual-pipeline-stage 4     # Virtual pipeline for load balancing

# CP
#--context-parallel-size 2           # 2-way context parallelism
#--cp-comm-type p2p                  # Communication type

# EP
#--expert-model-parallel-size 8  # 8-way expert parallelism
#--num-experts 64                # 64 experts per MoE layer
#--moe-grouped-gemm              # Optimize expert computation

