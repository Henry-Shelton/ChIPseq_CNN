#!/usr/bin/env bash
module load bioinformatics
module load sratoolkit/2.11.0
vdb-config --interactive # config sratoolkit, choose TOOLs->prefetch downloads to “current directory”, otherwise, the program will not work
cd /nobackup/$USER/PRJNA215956
if [ ! -d fastq ]
then
    mkdir fastq
fi

cd fastq
fasterq-dump #see usage of this command
fasterq-dump -p SRR955635  #download fastq files of one run

run_names=( SRR9556{36..38} )
for run in ${run_names[@]}
do
  echo $run
  fasterq-dump -p $run
done
