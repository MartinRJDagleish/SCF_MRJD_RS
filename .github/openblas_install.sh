#!/bin/bash -e
# Install OpenBLAS

# Following the guide from: https://github.com/bgeneto/build-install-compile-openblas

# 1. Install prerequisites
sudo apt-get install -y build-essential gfortran g++ git libgsl-dev -y

# 2. mkdir for OpenBLAS
OPENBLAS_DIR=/opt/openblas
sudo mkdir $OPENBLAS_DIR

# 3. Build and install openblas library from source
# 3.1. Clone the OpenBLAS repository
cd $HOME
git clone https://github.com/xianyi/OpenBLAS

# 3.2 Build and install OpenBLAS (multithreaded, non OPENMP version)
cd $HOME/OpenBLAS
export USE_THREAD=1
export NUM_THREADS=2 # CHANGED SCRIPT HERE FROM 64 TO 4
export DYNAMIC_ARCH=0
export NO_WARMUP=1
export BUILD_RELAPACK=0
export COMMON_OPT="-O2 -march=native"
export CFLAGS="-O2 -march=native"
export FCOMMON_OPT="-O2 -march=native"
export FCFLAGS="-O2 -march=native"
echo ""
echo "Building OpenBLAS with $NUM_THREADS threads"
echo ""
make -j DYNAMIC_ARCH=0 CC=gcc FC=gfortran HOSTCC=gcc BINARY=64 INTERFACE=64 LIBNAMESUFFIX=threaded \
sudo make PREFIX=$OPENBLAS_DIR LIBNAMESUFFIX=threaded install


# 4. Test install
echo "Testing OpenBLAS install"
make -j lapack-test
cd ./lapack-netlib; python3 ./lapack_testing.py -r -b TESTING

# 5. Install
sudo make install

export C_INCLUDE_PATH=$C_INCLUDE_PATH:/opt/OpenBLAS/include
export CPATH=$CPATH:/opt/OpenBLAS/include
export LIBRARY_PATH=$LIBRARY_PATH:/opt/OpenBLAS/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib

# 6 Compile and linking with openblas
# gcc -I/opt/openblas/include -pthread -O3 -Wall example.c -o ~/bin/example -L/opt/openblas/lib -lm -lpthread -lgfortran -lopenblas

# export MAX_THREADS=4
# export OPENBLAS_NUM_THREADS=$MAX_THREADS
# export GOTO_NUM_THREADS=$MAX_THREADS
# export OMP_NUM_THREADS=$MAX_THREADS
# export MKL_NUM_THREADS=$MAX_THREADS
# export BLIS_NUM_THREADS=$MAX_THREADS
