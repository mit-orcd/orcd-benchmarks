for dir in `cat dirs`
do
    rsync -aroguv $dir ../bak/benchmarks/
done
