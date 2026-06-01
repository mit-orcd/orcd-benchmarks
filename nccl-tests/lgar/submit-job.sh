#!/bin/bash

#sbatch -p pi_keating 1node.sh
#sbatch -p pi_laub 1node.sh
#sbatch -p pi_srmadden 1node.sh
#sbatch -p ou_sloan_teaching 1node.sh
#sbatch -p pi_ashia07 1node.sh
#sbatch -p mit_normal_gpu 1node.sh

#sbatch -p mit_normal_gpu -w node[2433-2434] 2nodes-2gpus.sh
#sbatch -p ou_bcs_low 2nodes-2gpus.sh
#sbatch -p pi_keating 2nodes-2gpus.sh

#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited 1node.sh

sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3206-3207] 2nodes-2gpus.sh
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3204-3205] 2nodes-2gpus.sh

#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3300,3400] 2nodes-2gpus.sh
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3101,3400] 2nodes-2gpus.sh
#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited -w node[3101,3000] 2nodes-2gpus.sh


#sbatch --reservation=orcd_testing -p mit_normal_gpu -q unlimited 2nodes-all-gpus.sh

# h200 nodes
#node3000,node3101,node3300,node3400
