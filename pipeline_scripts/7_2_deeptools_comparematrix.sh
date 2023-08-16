#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools_matrix                # This name will let you follow your job
#SBATCH --output=../log/deeptools_matrix%A_%a.out
#SBATCH --error=../log/deeptools_matrix%A_%a.err

#module import
module load bioinformatics
module load gcc
module load python
module load deeptools
module load samtools
module load bamtools


#deeptools coverage calculation
echo "compute matrix calculation..."
matrix_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/
cd ${matrix_dir}

#rep1_2
computeMatrix reference-point --referencePoint TSS \
 -S /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/1_bam_to_bigwig_coverage/test_rep1_ali.bw /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/1_bam_to_bigwig_coverage/test_rep2_ali.bw \
 -R /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_0_all_genes/chr19_feb09_UCSCgenes.bed \
 -b 5000 \
 -a 5000 \
 --skipZeros \
 -p 10 \
 -bl /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_1_blacklist/ENCFF356LFX.bed \
 -o /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/rep1_2_ali_TSS_matrix.gz \
 --outFileSortedRegions /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/rep1_2_ali_TSS_matrix_regions.bed

echo "coverage done: output matrix!" 
echo""