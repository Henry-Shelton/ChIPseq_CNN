
#!/bin/bash
#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 80                                  # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=100G                              # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:200G                         # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00                            # time limit in format dd-hh:mm:ss
#SBATCH --job-name=deeptools_matrix_norm                # This name will let you follow your job
#SBATCH --output=../log/deeptools_matrix_norm%A_%a.out
#SBATCH --error=../log/deeptools_matrix_norm%A_%a.err

#module import
module load bioinformatics
module load gcc
module load python
module load deeptools
module load samtools
module load bamtools


#deeptools coverage calculation
echo "bamCoverage calculation..."
matrix_dir=/nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_1_computeMatrix_NORM/
cd ${matrix_dir}

#rep1_2
computeMatrix reference-point --referencePoint TSS \
 -S /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/2_normalise_compare_bam_to_bigwig/normalised_rep1.bw /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/2_normalise_compare_bam_to_bigwig/normalised_rep2.bw \
 -R /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_0_all_genes/all_genes.bed \
 -bl /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/3_1_blacklist/ENCFF356LFX.bed \
 -b 1000 \
 -a 1000 \
 --binSize 25 \
 --skipZeros \
 -p 6 \
 -o /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_1_computeMatrix_NORM/matrix_normrep1_normrep2_TSS.gz \
 --outFileSortedRegions /nobackup/kfwc76/DISS/data/1_8_datavis_deeptools_bigwig/4_1_computeMatrix_NORM/normrep1_normrep2_TSS.bed

echo "coverage done: output matrix + regions!" 
echo""