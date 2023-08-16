#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 4                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=8G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:12G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=3_bowtie_alignment # This name will let you follow your job
#SBATCH --output=../log/3_bowtie_alignment%A_%a.out
#SBATCH --error=../log/3_bowtie_alignment%A_%a.err
#SBATCH --array=1-4

genome_dir=/nobackup/kfwc76/DISS/data/1_0_genome/
output_dir=/nobackup/kfwc76/DISS/data/1_3_bowtie_aligned/
#Check if the output_dir exists. If not, create one.
if [ ! -d $output_dir ]
then
    mkdir $output_dir
fi

module load bioinformatics
module load bowtie2

bowtie2 -x /nobackup/kfwc76/DISS/data/1_0_genome/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -U /nobackup/kfwc76/DISS/data/1_2_trimmed_files/ENCFF310LKV_1_val_1.fq --thread 1 -o $output_dir

bowtie2 -x /nobackup/kfwc76/DISS/data/1_0_genome/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta -U /nobackup/kfwc76/DISS/data/1_2_trimmed_files/ENCFF492XRZ_2_val_2.fq --thread 1 -o $output_dir
