
for i in 002 004 005 006 007 008 009 010 011 012 013
do
   file="/orcd/scratch/orcd/$i/shaohao/wikipedia_tokenized.pt"
   ls $file
   sbatch job-llama.sh $file
done

