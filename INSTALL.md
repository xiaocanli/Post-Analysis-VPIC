# Install Guide

## Overview
This package uses [FLAP](https://github.com/szaghi/FLAP) to deal with comand line arguments.The FLAP package is recommended to install using the [FoBiS](https://github.com/szaghi/FoBiS), which is building system for Fortran projects.

## Requirments
- **Git**. The newer the better.
- **CMake** 3.0.0 or higher
- **GCC** or **Intel** compilers. Not sure if it works for different versions.
- **OpenMPI** and **MVAPICH2**. Minimum versions are unknown.
- Parallel **HDF5**. Tested with version 1.8.13 and 1.8.16
- On a LANL cluster, `source module_intel_lanl.sh` to load the above packages. On a Cray cluster, `source module_cray.sh`
- [FoBiS](https://github.com/szaghi/FoBiS)
  - The installation wiki: https://github.com/szaghi/FoBiS/wiki/Install.
  - It is recommended to use PyPI to install it. Before installation, load a python module on a cluster.
  For example,`module load python/2.7-anaconda-4.1.1` on a LANL cluster.
  - Then, `pip install FoBiS.py --user`

## Download
In the root directory of PIC run,
```sh
$ git clone --recursive -b pfields https://github.com/xiaocanli/Post-Analysis-VPIC
$ mv Post-Analysis-VPIC pic_analysis
```
Don't ask me why I have to change the repository name after cloning it.

## Install
- We need to install [FLAP](https://github.com/szaghi/FLAP) first.
```sh
$ cd src/third_party/FLAP/
$ FoBiS.py build -mode static-intel
```
- In the top directory of a run,
```sh
$ mkdir build
$ cd build
$ cmake ..
$ make
$ make install
```
