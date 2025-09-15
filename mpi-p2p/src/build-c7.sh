# 1 = c7 or r8
# 2 = module name: openmpi, intel
# 3 = version

OSU_BENCH=osu-micro-benchmarks-7.5-1
DESTINATION="$1-$3"
INSTALL_DIR="$PWD/../install/$DESTINATION"

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

module load gcc/6.2.0
module load "$2/$3"  # openmpi/$2

if [ $2 = "intel" ]; then
  MPIC=mpiicx
  MPICPP=mpiicpx
else
  MPIC=mpicc
  MPICPP=mpicxx
fi

which $MPIC
which $MPICPP
$MPIC --version
$MPICPP --version

# build. No command should run in background with &
cd $DESTINATION
./configure CC=$MPIC CXX=$MPICPP --prefix=$INSTALL_DIR >log.config
make clean
make >log.make
make install >log.install
cd ..


