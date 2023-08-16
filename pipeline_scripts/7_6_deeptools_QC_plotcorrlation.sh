#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools_QC_Plot                # This name will let you follow your job
#SBATCH --output=../log/deeptools_QC_Plot%A_%a.out
#SBATCH --error=../log/deeptools_QC_Plot%A_%a.err

#module import
module load bioinformatics
module load gcc
module load deeptools

#plot deeptools QC
DT_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/5_deeptools_QC/
cd ${DT_dir}
plotCorrelation --corData rep1_2_deeptools_multiBAM.out.npz \
--plotFile deepTools_scatterplot.png \
--removeOutliers \
--whatToPlot scatterplot \
--corMethod pearson \
--labels Control Rep1 Rep2 