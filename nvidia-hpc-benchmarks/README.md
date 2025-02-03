## Download
First, downlad a container image from NGC. 
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks
```
module load apptainer
apptainer pull docker://nvcr.io/nvidia/hpc-benchmarks:24.09
```

## Submit the job
sbatch job.sh

