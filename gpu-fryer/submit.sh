for node in `cat h200.txt`; do
    echo $node
    sbatch -w $node --gres=gpu:h200:8 test.sh
done 

for node in `cat rtx6000.txt`; do
    sbatch -w $node --gres=gpu:rtx_pro_6000:8 test.sh
done 
