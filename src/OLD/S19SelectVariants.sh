#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 1            # number of CPU cores required, one per thread, up to 128
#SBATCH --mem=2G        # memory required, up to 250G.
#SBATCH --gres=tmp:3G   # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S19SelectVariants # This name will let you follow your job
#SBATCH --output=../log/S19SelectVariants%A_%a.out
#SBATCH --error=../log/S19SelectVariants%A_%a.err
#SBATCH --array=1

work_dir=/nobackup/$USER/PRJNA215956/work
sample_names=("" "BCF2")
sample=${sample_names[$SLURM_ARRAY_TASK_ID]}
cd ${work_dir}/${sample}_strelka/results/variants
ref_fa=/nobackup/$USER/PRJNA215956/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa

module load bioinformatics
module load gatk
gatk --java-options "-Xmx4g" SelectVariants \
-V somatic.snvs.vcf.gz \
--reference ${ref_fa} \
--exclude-filtered true \
-O somatic.snvs.pass.vcf

gatk --java-options "-Xmx4g" SelectVariants \
-V somatic.indels.vcf.gz \
--reference ${ref_fa} \
--exclude-filtered true \
-O somatic.indels.pass.vcf
