#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 4                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=8G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:12G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S06fastqc # This name will let you follow your job
#SBATCH --output=../log/1_fastqc%A_%a.out
#SBATCH --error=../log/1_fastqc%A_%a.err
#SBATCH --array=1-4

data_dir=/nobackup/kfwc76/DISS/data/1_2_trimmed_files/
output_dir=/nobackup/kfwc76/DISS/data/1_2_trimmed_files/

cd ${data_dir}
run_names_1=( "" ENCFF310LKV_1_val_1 )
run_name_1=${run_names_1[$SLURM_ARRAY_TASK_ID]}

run_names_2=( "" ENCFF492XRZ_2_val_2 )
run_name_2=${run_names_2[$SLURM_ARRAY_TASK_ID]}

module load bioinformatics
module load fastqc
fastqc -o $output_dir -t 4 ${run_name_1}.fq
fastqc -o $output_dir -t 4 ${run_name_2}.fq
