#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:05:00
#BB destroy_persistent name=reconnection
