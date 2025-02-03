
# 1 = c7 or r8
# 2 = module name

OSU_BENCH=osu-micro-benchmarks-7.3
DESTINATION="$1-nvhpc-$2"
INSTALL_DIR="/orcd/home/001/shaohao/mpi/osu-bench/install/$DESTINATION"

# create install dir
rm -rf $INSTALL_DIR
if [ ! -d $INSTALL_DIR ]; then
   mkdir -p $INSTALL_DIR
fi

# exact files
rm -rf $OSU_BENCH $DESTINATION 
if [ ! -d $OSU_BENCH ]; then
   tar xvf ${OSU_BENCH}.tar.gz 
   mv $OSU_BENCH $DESTINATION
fi

# load modules
module purge
module load nvhpc/2023_233/nvhpc/$2

which mpicc
mpicc --version

# build. No command should run in background with &
cd $DESTINATION
./configure CC=mpicc CXX=mpicxx --prefix=$INSTALL_DIR >log.config
make clean
make >log.make
make install >log.install
cd ..
