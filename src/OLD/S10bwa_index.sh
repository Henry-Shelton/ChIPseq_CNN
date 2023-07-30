#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 2                # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=4G            # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:6G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S10bwa_index # This name will let you follow your job
#SBATCH --output=../log/S10bwa_index%A_%a.out
#SBATCH --error=../log/S10bwa_index%A_%a.err
#SBATCH --array=1

module load bioinformatics
module load bwa-mem2

cd /nobackup/$USER/PRJNA215956/genome
if [ ! -d bwa ]
then
  mkdir bwa
fi
cd bwa
if [ ! -f TAIR10.fa ]
then
  ln -s /nobackup/prsd58/PRJNA215956/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa TAIR10.fa
fi
bwa-mem2 index TAIR10.fa
