#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools                # This name will let you follow your job
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
echo "bamCoverage calculation..."
matrix_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_computeMatrix_matrix_regions/
cd ${matrix_dir}

#rep1_2
computeMatrix reference-point --referencePoint TSS \
 -S /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/Main_outputs_for_Replicates_1_2/fold_change_over_control.bigWig \
 -R /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_0_all_genes/all2.bed \
 -bl /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_1_blacklist/ENCFF356LFX.bed \
 -b 1000 \
 -a 1000 \
 --binSize 20 \
 --skipZeros \
 -o /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/matrix_rep1_rep2_TSS_chr12.gz \
 --outFileSortedRegions /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/rep_1_rep_2_regions_TSS_chr12.bed

echo "coverage done: output matrix!" 
echo""


