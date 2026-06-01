image=$1
#gpu_count=$2

sing_cmd="singularity exec --nv $image"
N_mat=230000 # 247300 # max usage of H200 memory, 100000
N_3D=256

#echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run hpl and hpl-mxp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# run a single gpu job
#echo "======================= 1 GPU ========================="
#$sing_cmd /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
#$sing_cmd /workspace/hpl-mxp.sh --n $N_mat --nb 2048 --nprow 1 --npcol 1 --nporder row --gpu-affinity 0

# run 2 gpu job if possible
#if [ $gpu_count -ge 2 ]; then
#echo "======================= 2 GPUs ========================="
#  $sing_cmd mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
#  $sing_cmd mpirun -np 2 /workspace/hpl-mxp.sh --n $N_mat --nb 2048 --nprow 1 --npcol 2 --nporder row --gpu-affinity 0:1
#fi

# run 4 gpu job if possible
#if [ $gpu_count -ge 4 ]; then
#  echo "====================== 4 GPUs ========================="
#  $sing_cmd mpirun -np 4 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
#  $sing_cmd mpirun -np 4 /workspace/hpl-mxp.sh --n $N_mat --nb 2048 --nprow 2 --npcol 2 --nporder row --gpu-affinity 0:1:2:3
#fi

# run 8 gpu job if possible
#if [ $gpu_count -ge 8 ]; then
#  echo "====================== 8 GPUs ========================="
#  $sing_cmd mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat
#  $sing_cmd mpirun -np 8 /workspace/hpl-mxp.sh --n $N_mat --nb 2048 --nprow 2 --npcol 4 --nporder row --gpu-affinity 0:1:2:3:4:5:6:7
#fi

#echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run hpcg %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
#$sing_cmd /workspace/hpcg.sh --nx $N_3D --ny $N_3D --nz $N_3D --rt 120


echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run stream %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
array size = 430080000 is suggested to comparte with AMD CPUs
#array size = 53760000 is for an instant of 18 GB
#size=10000
# loop through all GPUsa, only one instant
  echo "========= stream of instant ==========="
  $sing_cmd /workspace/stream-gpu-linux-x86_64/stream-gpu-test.sh -n $size -d 0

