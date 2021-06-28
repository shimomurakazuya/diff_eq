#!/bin/bash
#PBS -q pg1
#PBS -l select=1:ncpus=48:mpiprocs=4:ompthreads=12:ngpus=4
#PBS -P job
#PBS -l walltime=01:30:00
#PBS -N dif2d 

#. /etc/profile.d/modules.sh

# module load intel/cur mpt/cur

DIR=data

cd $PBS_O_WORKDIR

mkdir -p $DIR
export MPI_SHEPHERD=1

echo date


#dplace ./run > ouput. log 2>&1    
#./run |& tee log.txt
./run 


