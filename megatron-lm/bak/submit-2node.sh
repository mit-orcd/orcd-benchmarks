#sbatch --gpus-per-node=l40s:1  2node.sh
sbatch --gpus-per-node=l40s:2  2node.sh
sbatch --gpus-per-node=l40s:4  2node.sh
sbatch --gpus-per-node=h200:1  2node.sh
sbatch --gpus-per-node=h200:2  2node.sh
sbatch --gpus-per-node=h200:4  2node.sh
sbatch --gpus-per-node=h200:8  2node.sh
