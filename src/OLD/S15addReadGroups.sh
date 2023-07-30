#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 2                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=4G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:3G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S15addReadGroups # This name will let you follow your job
#SBATCH --output=../log/S15addReadGroups%A_%a.out
#SBATCH --error=../log/S15addReadGroups%A_%a.err
#SBATCH --array=1-2
work_dir=/nobackup/$USER/PRJNA215956/work
cd ${work_dir}
sample_names=("" "mir159a" "BCF2")
sample=${sample_names[$SLURM_ARRAY_TASK_ID]}

module load bioinformatics
module load gatk
gatk --java-options "-Xmx4g"  AddOrReplaceReadGroups \
-I ${sample}.sorted.bam \
-O ${sample}.RG.bam \
--RGLB ${sample} \
--RGPL illumina \
--RGPU ${sample} \
--RGSM ${sample} \
--RGID ${sample} \
--SORT_ORDER coordinate \
--CREATE_INDEX true \
--TMP_DIR ${sample}tmp
