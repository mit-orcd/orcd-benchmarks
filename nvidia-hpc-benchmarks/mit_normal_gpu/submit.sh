#sbatch --reservation=mapo-2025 -p pi_linaresr --gres=gpu:h100:8 1node.sh
#sbatch --reservation=mapo-2025 -p pi_ppliang --gres=gpu:h200:8 1node.sh

#sbatch -p pi_linaresr --gres=gpu:h100:8 1node.sh
sbatch -p mit_normal_gpu -q unlimited  --gres=gpu:h200:8 1node.sh
