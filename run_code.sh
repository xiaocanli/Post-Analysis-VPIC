#!/bin/bash

cd build_intel
make
make install
cd ..

mpirun -np 128 ./parallel_hdf5
# mpirun -np 64 ./translate
# mpirun -np 64 ./dissipation -s e
