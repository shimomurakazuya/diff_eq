#!/bin/bash
#PBS -q pg1
#PBS -l select=1:ncpus=48:mpiprocs=4:ompthreads=12:ngpus=4
#PBS -P job
#PBS -l walltime=01:30:00
#PBS -N dif2d 

#. /etc/profile.d/modules.sh

# module load intel/cur mpt/cur

cd $PBS_O_WORKDIR

export MPI_SHEPHERD=1

echo date
#dplace ./run > ouput. log 2>&1    
./run

###$ qsub -I -q pg1 -l select=1:ncpus=48:mpiprocs=4:ompthreads=12:ngpus=4 -l walltime=24:0:0 -P interactive -N interactive
###qsub: waiting for job 197667.s86pbs01 to start
###qsub: job 197667.s86pbs01 ready
###
###$ cd $PBS_O_WORKDIR
###
###$ ./run
###nx = 2000000, ny= 96, iter = 20, 
###Segmentation fault

