#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 4                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=8G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:12G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S06fastqc # This name will let you follow your job
#SBATCH --output=../log/S06fastqc%A_%a.out
#SBATCH --error=../log/S06fastqc%A_%a.err
#SBATCH --array=1-4

data_dir=/nobackup/$USER/PRJNA215956/fastq
output_dir=/nobackup/$USER/PRJNA215956/fastqc
#Check if the output_dir exists. If not, create one.
if [ ! -d $output_dir ]
then
    mkdir $output_dir
fi

cd ${data_dir}
run_names=( "" SRR9556{35..38} )
run_name=${run_names[$SLURM_ARRAY_TASK_ID]}

module load bioinformatics
module load fastqc
fastqc -o $output_dir -t 4 ${run_name}_1.fastq ${run_name}_2.fastq

#See section 5.1 for more details
