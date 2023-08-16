#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools                # This name will let you follow your job
#SBATCH --output=../log/deeptools_plot%A_%a.out
#SBATCH --error=../log/deeptools_plot%A_%a.err

#module import
module load bioinformatics
module load gcc
module load python
module load deeptools
module load samtools
module load bamtools

#plot profile 

matrix_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/
cd ${matrix_dir}
plotHeatmap -m /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/rep1_2_ali_TSS_matrix.gz \
-out /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_computeMatrix_matrix_regions/rep1_2_ali_TTS_heatmap.png \