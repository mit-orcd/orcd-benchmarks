#!/bin/bash

sbatch -p pi_rteague -w node5106 --gres=gpu:2 1node.sh 2
sbatch -p ou_cheme_gpu --gres=gpu:rtx_pro_6000:8 1node.sh 8
 sbatch -p pi_lauffen --gres=gpu:2 1node.sh 2


#sbatch -p pi_keating 1node.sh
#sbatch -p pi_laub 1node.sh
#sbatch -p pi_srmadden 1node.sh
#sbatch -p ou_sloan_teaching 1node.sh
#sbatch -p pi_ashia07 1node.sh
#sbatch -p mit_normal_gpu 1node.sh
#sbatch -p pi_jhm --reservation=orcd_testing --gres=gpu:4 1node.sh

#sbatch -p mit_normal_gpu -w node[2433-2434] 2nodes-2gpus.sh
#sbatch -p ou_bcs_low 2nodes-2gpus.sh
#sbatch -p pi_keating 2nodes-2gpus.sh
#sbatch -p ou_bcs_low --gpus-per-node=1 2nodes-2gpus.sh

#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited 1node.sh
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node4208 1node.sh 4
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node3301 --gres=gpu:h200:8 1node.sh
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node3200 --gres=gpu:h200:8 1node.sh 8
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node3201 --gres=gpu:h200:8 1node.sh 8
#sbatch --reservation=orcd_testing -p ou_sloan_gpu -w node4400 --gres=gpu:h200:8 1node.sh 8

#sbatch --reservation=mapo-2025 -p mit_normal_gpu -q unlimited --gres=gpu:h200:8 1node.sh 8
#sbatch --reservation=mapo-2025 -p mit_normal_gpu -q unlimited --gres=gpu:4 1node.sh 4
#sbatch --reservation=mapo-2025 -p pi_ppliang --gres=gpu:h200:8 1node.sh 8
#sbatch --reservation=mapo-2025 -p pi_ppliang --gres=gpu:h200:4 1node.sh 4
#sbatch --reservation=mapo-2025 -p pi_ccoley --gres=gpu:h100:4 1node.sh 4
#sbatch --reservation=mapo-2025 -p ou_bcs_low --gres=gpu:h100:4 1node.sh 4
#sbatch -p pi_linaresr --gres=gpu:h100:8 1node.sh 8

#sbatch --reservation=orcd_testing -p pi_melkin 1node.sh 4

#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3206-3207] 2nodes-2gpus.sh
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3204-3205] 2nodes-2gpus.sh
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3404,3506] 2nodes-2gpus.sh
# sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3200,3201] --gpus-per-node=h200:1 2nodes-2gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -w node[3100,3101] 2nodes-2gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -w node[3100,3200] 2nodes-2gpus.sh
sbatch -p mit_normal_gpu -q unlimited -w node[3100,3201] 2nodes-2gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -w node[3101,3200] 2nodes-2gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -x node3100 2nodes-2gpus.sh

#sbatch --reservation=mapo-2025 -p mit_normal_gpu -q unlimited --gpus-per-node=h200:1 2nodes-2gpus.sh
#sbatch --reservation=mapo-2025 -p mit_normal_gpu -q unlimited --gpus-per-node=1 2nodes-2gpus.sh

#sbatch -p mit_normal_gpu -q unlimited -w node[3405,3406] --gpus-per-node=1 2nodes-2gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -w node[3301,3400] --gpus-per-node=h200:1 2nodes-2gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -w node[2433,2434] --gpus-per-node=h200:1 2nodes-2gpus.sh


# ===================== all GPUs ==================
#sbatch -p mit_normal_gpu -q unlimited -w node[2433,2434] --gpus-per-task=h200:1 2nodes-all-gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -w node[2433,2434] --gpus-per-node=h200:4 2nodes-all-gpus.sh
#sbatch -p mit_normal_gpu -q unlimited -w node[3300,3301] --ntasks-per-node=8 --gpus-per-node=h200:8 2nodes-all-gpus.sh

#sbatch -p mit_normal_gpu -q unlimited --ntasks-per-node=4 --gpus-per-node=h200:4 2nodes-all-gpus.sh

