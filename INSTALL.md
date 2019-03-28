# Install Guide

## Overview
This package uses [FLAP](https://github.com/szaghi/FLAP) to deal with comand line arguments.The FLAP package is recommended to be installed using the [FoBiS](https://github.com/szaghi/FoBiS) package, which is a building system for Fortran projects.

## Requirments
- **Git** The newer the better.
- **CMake** 3.0.0 or higher
- **GCC** or **Intel** compilers. Not sure if it works for different versions.
    1. Failed compiling with intel/18.0.5 (reason unknown).
- **OpenMPI** and **MVAPICH2**. Minimum versions are unknown.
- Parallel **HDF5**. Tested with version 1.8.13 and 1.8.16
- [FoBiS](https://github.com/szaghi/FoBiS)
  - The installation wiki: https://github.com/szaghi/FoBiS/wiki/Install.
  - It is recommended to use PyPI to install it. Before installation, load a python module on a cluster.
  For example,`module config_files/load python/2.7-anaconda-4.1.1` on a LANL cluster.
  - Then, `pip install FoBiS.py --user`

## Download
In the root directory of PIC run,
```sh
$ git clone --recursive https://github.com/xiaocanli/Post-Analysis-VPIC
$ mv Post-Analysis-VPIC pic_analysis
```
Don't ask me why I have to change the repository name after cloning it.

## Install
- On a LANL cluster, `source config_files/module_intel_lanl.sh` to load above packages. On a Cray cluster, `source config_files/module_cray.sh`.

- In the directory `pic_analysis`,
```sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ make install
```
- On a cray system, the above `cmake ..` should be
```sh
CC=cc CXX=CC FC=ftn cmake ..
```
