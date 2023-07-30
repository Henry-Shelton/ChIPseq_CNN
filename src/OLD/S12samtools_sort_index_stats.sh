#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 32                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=64G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:96G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S12samtools_sort_index_stats.sh # This name will let you follow your job
#SBATCH --output=../log/S12samtools_sort_index_stats%A_%a.out
#SBATCH --error=../log/S12samtools_sort_index_stats%A_%a.err
#SBATCH --array=1-2
module load bioinformatics
module load samtools
work_dir=/nobackup/$USER/PRJNA215956/work
cd ${work_dir}
sample_names=("" "mir159a" "BCF2")

sample=${sample_names[$SLURM_ARRAY_TASK_ID]}

samtools sort -m 8G \
${sample}.sam \
-T ${sample} \
-o ${sample}.sorted.bam \
--threads 8 && \

samtools index \
${sample}.sorted.bam && \

samtools stats \
--threads 16 \
${sample}.sorted.bam \
> ${sample}_sorted_bam_stats.txt
