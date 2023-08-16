#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools                # This name will let you follow your job
#SBATCH --output=../log/deeptools_plot_NORM%A_%a.out
#SBATCH --error=../log/deeptools_plot_NORM%A_%a.err

#module import
module load bioinformatics
module load gcc
module load python
module load deeptools
module load samtools
module load bamtools

#plot profile 

matrix_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_1_computeMatrix_NORM/
cd ${matrix_dir}
plotProfile -m /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_1_computeMatrix_NORM/matrix_normrep1_normrep2_TSS.gz \
 -out /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_1_computeMatrix_NORM/matrix_normrep1_normrep2_TSS_profile.png \
 --perGroup \
 --colors green purple \
 --plotTitle "" --samplesLabel "Rep1" "Rep2" \
 --refPointLabel "TSS" \
 -T "Control read density" \
 -z ""
