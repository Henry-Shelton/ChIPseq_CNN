#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 20                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=40G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:12G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=trim_galore                 # This name will let you follow your job
#SBATCH --output=../log/2_trim%A_%a.out
#SBATCH --error=../log/2_trim%A_%a.err
#SBATCH --array=1-4

data_dir=/nobackup/$USER/DISS/data/fastq
output_dir=/nobackup/$USER/DISS/data/trimmed_files
#Check if the output_dir exists. If not, create one.
if [ ! -d $output_dir ]
then
    mkdir $output_dir
fi

module load bioinformatics
module load fastqc
module load trim_galore

trim_galore --paired --q 30 --gzip --fastqc /nobackup/$USER/DISS/data/fastq/ENCFF310LKV_1.fastq /nobackup/$USER/DISS/data/fastq/ENCFF492XRZ_2.fastq -o $output_dir
