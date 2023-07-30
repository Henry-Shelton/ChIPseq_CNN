#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=motif_prep                # This name will let you follow your job
#SBATCH --output=../log/motif_prep%A_%a.out
#SBATCH --error=../log/motif_prep%A_%a.err

#module import
module load bioinformatics
module load bedtools
module load gcc

#motif discovery

motif_dir=/nobackup/kfwc76/DISS/data/1_9_R_ChIPseq/combined_motif_discovery/
cd ${motif_dir}
bedtools getfasta -fi \
/nobackup/kfwc76/DISS/data/1_9_R_ChIPseq/combined_motif_discovery/hg38.fa \
-bed combinedrep-idr-merged-simple.bed \
-fo combinedreps-idr-merged-dreme.fasta
