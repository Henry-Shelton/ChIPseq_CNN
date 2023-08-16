#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --gres=tmp:12G
#SBATCH -t 01:00:00
#SBATCH --job-name=samtools_sort_index
#SBATCH --output=../log/samtools_sort_index%A_%a.out
#SBATCH --error=../log/samtools_sort_index%A_%a.err

#module import
module load bioinformatics
module load samtools

#samtools sorting 
echo "samtools sorting aligned bam files..."
bam_dir=/nobackup/kfwc76/DISS/data/1_4_samtools_bam/aligned_bam
cd ${bam_dir}
samtools sort -n "rep_1_aligned.bam" > "rep_1_aligned_sorted.bam"
samtools sort -n "rep_2_aligned.bam" > "rep_2_aligned_sorted.bam"
echo "sorting success! aligned.sorted" 
echo""

#samtools indexing 
echo "samtools indexing sorted aligned bam files..."
samtools index rep_1_aligned_sorted.bam
samtools index rep_2_aligned_sorted.bam
echo "sorting success! aligned.sorted" 
echo""

