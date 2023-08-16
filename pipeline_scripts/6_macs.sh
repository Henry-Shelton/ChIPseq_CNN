#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 20                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=40G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:12G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=macs2                # This name will let you follow your job
#SBATCH --output=../log/macs2%A_%a.out
#SBATCH --error=../log/macs2%A_%a.err

#module import
module load bioinformatics
module load macs
module load macs2

#macs2 peak enrichment relative to control
echo "macs2 peak enrichment relative to control calculation..."
bam_dir=/nobackup/kfwc76/DISS/data/1_4_samtools_macs_bam/macs2_enrichment/
cd ${bam_dir}

macs2 callpeak -t /nobackup/kfwc76/DISS/data/1_4_samtools_macs_bam/samtools_multi_bam_summary/rep_2_ali.bam -c /nobackup/kfwc76/DISS/data/1_4_samtools_macs_bam/0_control_bam/aligned_control.bam

echo "macs2 peak enrichment calculated!" 
echo""