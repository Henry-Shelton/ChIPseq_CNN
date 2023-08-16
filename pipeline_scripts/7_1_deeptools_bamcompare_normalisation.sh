#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools_bamcompare               # This name will let you follow your job
#SBATCH --output=../log/deeptools_bamcompare%A_%a.out
#SBATCH --error=../log/deeptools_bamcompare%A_%a.err

#module import
module load bioinformatics
module load deeptools/3.5.2

#deeptools coverage calculation
echo "bamCoverage calculation..."
bw_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/1_bam_to_bigwig_coverage/
cd ${bw_dir}

bamCompare -b1 /nobackup/kfwc76/DISS/data/1_4_samtools_macs_bam/samtools_multi_bam_summary/rep__ali.bam \
  -b2 /nobackup/kfwc76/DISS/data/1_4_samtools_macs_bam/0_control_bam/aligned_control.bam \
  --binSize 20 \
  --normalizeUsing BPM \
  --scaleFactorsMethod None \
  --smoothLength 60 \
  --extendReads 150 \
  --centerReads \
  -p 10 \
  -o /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/2_normalise_compare_bam_to_bigwig/normalised_rep2.bw

echo "coverage done: output results_dir matrix!" 
echo""

#this takes around 1 hour

