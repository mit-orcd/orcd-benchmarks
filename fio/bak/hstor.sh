#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=20GB
#SBATCH -p ou_bcs_low
#SBATCH -J hstor
#SBATCH -o output/out.%x-%N-%J

DIR=/orcd/data/orcd/001/io/fio-data/
mkdir -p $DIR
RW=read  # read # write
N_JOBS=16
NO_CACHE=1
ENGINE=psync
FNAME=data

echo "Data directory: $DIR"
echo "Read or write: $RW"
echo "Number of jobs = $N_JOBS"
echo "Do not use cache: $NO_CACHE"

echo "==============================="

fio --name=$FNAME --rw=$RW --numjobs=$N_JOBS --direct=$NO_CACHE --directory=$DIR --size=10G --time_based --runtime=60s --ramp_time=2s --verify=0 --bs=1M --iodepth=64 --group_reporting=1 --ioengine=$ENGINE



