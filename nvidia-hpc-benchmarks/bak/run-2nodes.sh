
module load  apptainer/1.1.7-x86_64  squashfuse/0.1.104-x86_64

module list
which singularity

#singularity exec --nv $PWD/hpc-benchmarks_24.03.sif /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-1GPU.dat
#singularity exec --nv $PWD/hpc-benchmarks_24.03.sif mpirun -np 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-2GPUs.dat
singularity exec --nv $PWD/hpc-benchmarks_24.03.sif mpirun -npernode 2 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-4GPUs.dat
#singularity exec --nv $PWD/hpc-benchmarks_24.03.sif mpirun -np 8 /workspace/hpl.sh --dat /workspace/hpl-linux-x86_64/sample-dat/HPL-8GPUs.dat

