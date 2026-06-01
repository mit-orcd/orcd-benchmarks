#!/bin/bash
#sbatch --reservation=mapo-2025 -p ou_bcs_low --gres=gpu:h100:4 1node.sh 4
#sbatch --reservation=orcd_testing -p pi_melkin 1node.sh 4

sbatch --reservation=orcd_testing -p pi_mghassem --gres=gpu:10 -w node3800 1node.sh 10
sbatch --reservation=orcd_testing -p pi_mghassem --gres=gpu:10 -w node3801 1node.sh 10
sbatch --reservation=orcd_testing -p pi_mghassem --gres=gpu:7 -w node3802 1node.sh 7
sbatch --reservation=orcd_testing -p pi_mghassem --gres=gpu:8 -w node3900 1node.sh 8
sbatch --reservation=orcd_testing -p pi_mghassem --gres=gpu:5 -w node3901 1node.sh 5
sbatch --reservation=orcd_testing -p pi_mghassem --gres=gpu:9 -w node3902 1node.sh 9


