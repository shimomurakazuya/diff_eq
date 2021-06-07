cat>./run.sh  
#!/bin/bash 
#PBS -q pc2
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=4
##PBS -P job
#PBS -l walltime=0:10:00

 cd $PBS_O_WORKDIR
 . /etc/profile.d/modules.sh
#module load intel/cur  mpt/cur
module purge                                                                                          |~                                                                                                        
 module load mpt/2.23-ga cuda/11.0; 
 export MPI_SHEPHERD=1
 
 ./run > ./output.log 2>&1

qsub run.sh
2021.s86pbs01
