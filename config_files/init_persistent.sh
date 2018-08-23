#!/bin/bash
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:05:00
#BB create_persistent name=reconnection capacity=204800GB access_mode=striped type=scratch 
