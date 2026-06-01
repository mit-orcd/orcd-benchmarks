
# 1 = c7 or r8
# 2 = module name: nvhpc
# 3 = version

OSU_BENCH=osu-micro-benchmarks-7.5.2

DESTINATION="$1-$2-$3"

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

# load modules

module load "$2/$3"  # openmpi/$2

MPIC=mpicc
MPICPP=mpicxx

which nvc
which $MPIC
which $MPICPP
$MPIC --version
$MPICPP --version
$MPIC --show

# build. No command should run in background with &
cd $DESTINATION
echo "----- configure ------"
./configure CC=$MPIC CXX=$MPICPP --prefix=$INSTALL_DIR --enable-cuda --with-cuda=${NVHPC_ROOT}/cuda  >log.config
echo "----- clean ------"
make clean
echo "----- make ------"
make >log.make
echo "----- make install ------"
make install >log.install
cd ..


