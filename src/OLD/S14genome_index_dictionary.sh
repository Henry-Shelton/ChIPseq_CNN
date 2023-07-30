#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 1                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=2G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:3G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S14genome_index_dictionary # This name will let you follow your job
#SBATCH --output=../log/S14genome_index_dictionary%A_%a.out
#SBATCH --error=../log/S14genome_index_dictionary%A_%a.err
#SBATCH --array=1

cd /nobackup/$USER/PRJNA215956/genome

module load bioinformatics
module load gatk
gatk --java-options "-Xmx2g" CreateSequenceDictionary \
--REFERENCE Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
--OUTPUT Arabidopsis_thaliana.TAIR10.dna.toplevel.dict

module load samtools
samtools faidx Arabidopsis_thaliana.TAIR10.dna.toplevel.fa
