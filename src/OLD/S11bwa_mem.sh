#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 16                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=32G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:48G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S11bwa_mem # This name will let you follow your job
#SBATCH --output=../log/S11bwa_mem%A_%a.out
#SBATCH --error=../log/S11bwa_mem%A_%a.err
#SBATCH --array=1-2
module load bioinformatics
module load bwa-mem2
data_dir=/nobackup/$USER/PRJNA215956/fastq
work_dir=/nobackup/$USER/PRJNA215956/work
if [ ! -d $work_dir ]
then
  mkdir $work_dir
fi
cd ${work_dir}
sample_names=("" "mir159a" "BCF2")
sample=${sample_names[$SLURM_ARRAY_TASK_ID]}
refgenome=/nobackup/$USER/PRJNA215956/genome/bwa/TAIR10.fa

bwa-mem2 mem \
-t 16 \
${refgenome} \
$data_dir/${sample}_1.fastq \
$data_dir/${sample}_2.fastq \
>${sample}.sam
