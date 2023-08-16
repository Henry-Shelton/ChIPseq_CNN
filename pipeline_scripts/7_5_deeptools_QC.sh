#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptoolsQC              # This name will let you follow your job
#SBATCH --output=../log/deeptools_QC%A_%a.out
#SBATCH --error=../log/deeptools_QC%A_%a.err

#module import
module load bioinformatics
module load gcc
module load deeptools

#deeptools QC - Calc read coverage scores using multiBamSummary
DT_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/5_deeptools_QC/
cd ${DT_dir}
multiBamSummary bins --ignoreDuplicates -p 6 \
--bamfiles /nobackup/kfwc76/DISS/data/1_4_samtools_macs_bam/ACTUAL_ALI_BAI_REPS/*ali.bam \
-out rep1_2_deeptools_multiBAM.out.npz \
--outRawCounts readCounts.tab
