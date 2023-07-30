#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 16                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=32G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:48G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S16markdup # This name will let you follow your job
#SBATCH --output=../log/S16markdup%A_%a.out
#SBATCH --error=../log/S16markdup%A_%a.err
#SBATCH --array=1-2
work_dir=/nobackup/$USER/PRJNA215956/work
cd ${work_dir}
sample_names=("" "mir159a" "BCF2")

sample=${sample_names[$SLURM_ARRAY_TASK_ID]}

module load bioinformatics
module load gatk

gatk --java-options "-Xmx32g" MarkDuplicates \
--REMOVE_DUPLICATES true \
--INPUT ${sample}.RG.bam \
--OUTPUT ${sample}.markdup.bam \
--METRICS_FILE ${sample}.markdup.metrics \
--CREATE_INDEX true \
--TMP_DIR ${sample}tmp && \

module load samtools
samtools stats \
--threads 16 \
${sample}.markdup.bam \
> ${sample}.markdup.bam_stats.txt
