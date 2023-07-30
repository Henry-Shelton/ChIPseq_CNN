#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 2                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=4G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:6G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S17mosdepth # This name will let you follow your job
#SBATCH --output=../log/S17mosdepth%A_%a.out
#SBATCH --error=../log/S17mosdepth%A_%a.err
#SBATCH --array=1-2

module load bioinformatics
module load mosdepth

work_dir=/nobackup/$USER/PRJNA215956/work
cd ${work_dir}
sample_names=("" "mir159a" "BCF2")

sample=${sample_names[$SLURM_ARRAY_TASK_ID]}

mosdepth -t 2 \
--no-per-base \
${sample} \
${sample}.markdup.bam
