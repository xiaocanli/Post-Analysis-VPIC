#!/bin/bash

cd build
make
make install
cd ..

mpirun -np 64 ./translate
mpirun -np 64 ./dissipation -s e
