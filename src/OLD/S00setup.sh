#!/usr/bin/bash
cd /nobackup/$USER
project_dir=PRJNA215956
if [ ! -d $project_dir ]
then
    mkdir $project_dir
    cd $project_dir
    mkdir log
    mkdir fastq
    mkdir fastqc # for fastqc output
    mkdir multiqc_report
    mkdir result
    cp /nobackup/prsd58/PRJNA215956/src  ./
fi
