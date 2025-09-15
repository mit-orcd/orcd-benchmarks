
#sbatch -p mit_normal --array=1-10 job.sh scratch 005
#sbatch -p mit_normal_gpu --array=1-10 job.sh scratch 005
#sbatch -p mit_normal_gpu -w node3101 job.sh scratch 008
#sbatch -p mit_normal_gpu -w node3005 job.sh scratch 007
#sbatch -p mit_normal_gpu -w node3007 job.sh scratch 006
sbatch -p mit_normal job.sh scratch 002

