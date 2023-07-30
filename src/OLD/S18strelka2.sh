#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 32                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=64G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:96G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 48:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S18strelka2 # This name will let you follow your job
#SBATCH --output=../log/S18strelka2%A_%a.out
#SBATCH --error=../log/S18strelka2%A_%a.err
#SBATCH --array=1

work_dir=/nobackup/$USER/PRJNA215956/work
cd ${work_dir}
sample_names=("" "BCF2")
control_names=("" "mir159a" )
control_sample=${control_names[$SLURM_ARRAY_TASK_ID]}
mutant_sample=${sample_names[$SLURM_ARRAY_TASK_ID]}
ref_fa=/nobackup/$USER/PRJNA215956/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa

module load bioinformatics
module load strelka2
configureStrelkaSomaticWorkflow.py \
    --normalBam ${control_sample}.markdup.bam \
    --tumorBam ${mutant_sample}.markdup.bam \
    --ref ${ref_fa} \
    --runDir ${work_dir}/${mutant_sample}_strelka && \
# execution on a single local machine with 16 parallel jobs
${work_dir}/${mutant_sample}_strelka/runWorkflow.py -m local -j 16
