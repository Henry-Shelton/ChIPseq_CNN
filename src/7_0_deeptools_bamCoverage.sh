#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools_bamcoverage                # This name will let you follow your job
#SBATCH --output=../log/deeptools_bamcoverage%A_%a.out
#SBATCH --error=../log/deeptools_bamcoverage%A_%a.err

#module import
module load bioinformatics
module load samtools
module load deeptools/3.5.2

#deeptools coverage calculation
echo "bamCoverage calculation..."
bw_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/bam_to_bigwig_coverage/
cd ${bw_dir}

#for each rep + control
bamCoverage -b /nobackup/kfwc76/DISS/data/1_4_samtools_macs_bam/ACTUAL_ALI_BAI_REPS/rep2_ali.bam \
-o /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/1_bam_to_bigwig_coverage/test_rep2_ali.bw \
--binSize 30 \
--normalizeUsing BPM \
--smoothLength 200 \
--extendReads 150 \
--exactScaling \
--centerReads \
--blackListFileName /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_1_blacklist/ENCFF356LFX.bed \
--numberOfProcessors 10

echo "coverage done: output .bw" 
echo""

#this takes around 1 hour

