#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 1            # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=2G        # memory required, up to 250G.
#SBATCH --gres=tmp:3G   # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S20snpEff # This name will let you follow your job
#SBATCH --output=../log/S20snpEff%A_%a.out
#SBATCH --error=../log/S20snpEff%A_%a.err
#SBATCH --array=1
module load bioinformatics
module load snpeff/5.0
work_dir=/nobackup/$USER/PRJNA215956/work
sample_names=("" "BCF2")
sample=${sample_names[$SLURM_ARRAY_TASK_ID]}

input_data_dir=${work_dir}/${sample}_strelka/results/variants
cd $input_data_dir
mkdir snpEff_snvs
mkdir snpEff_indels
cd ${work_dir}/${sample}_strelka/results/variants/snpEff_snvs
java -Xmx4G -jar /nobackup/dbl0hpc/apps/snpEff/snpEff.jar \
  -c /nobackup/dbl0hpc/apps/snpEff/snpEff.config \
  Arabidopsis_thaliana.TAIR10.56  \
  ${input_data_dir}/somatic.snvs.pass.vcf > somatic.snvs.pass.ann.vcf

cd ${work_dir}/${sample}_strelka/results/variants/snpEff_indels
java -Xmx4G -jar /nobackup/dbl0hpc/apps/snpEff/snpEff.jar \
  -c /nobackup/dbl0hpc/apps/snpEff/snpEff.config \
  Arabidopsis_thaliana.TAIR10.56  \
  ${input_data_dir}/somatic.indels.pass.vcf > somatic.indels.pass.ann.vcf
