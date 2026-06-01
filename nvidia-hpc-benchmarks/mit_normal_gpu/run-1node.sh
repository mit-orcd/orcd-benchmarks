sbatch -p $1 1node.sh
#sbatch --reservation=orcd_testing  -p mit_normal_gpu -q unlimited -w $1 1node.sh

# sbatch --reservation=orcd_testing  -p $1 1node.sh
# sbatch --reservation=orcd_testing  -p mit_normal_gpu -w $1 1node.sh
#sbatch --reservation=orcd_testing  -p mit_normal_gpu 1node.sh
#sbatch --reservation=orcd_testing -q unlimited -p mit_normal_gpu 1node.sh
