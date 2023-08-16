#!/bin/bash
#SBATCH -p shared
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --gres=tmp:12G
#SBATCH -t 01:00:00
#SBATCH --job-name=dl_fastqc_multiqc
#SBATCH --output=../log/1_dl_fastqc_multiqc%A_%a.out
#SBATCH --error=../log/1_dl_fastqc_multiqc%A_%a.err

#module import
module load bioinformatics
module load fastqc

#data links
echo "initialising..."
rep_1_URL='https://www.encodeproject.org/files/ENCFF310LKV/@@download/ENCFF310LKV.fastq.gz'
rep_2_URL='https://www.encodeproject.org/files/ENCFF492XRZ/@@download/ENCFF492XRZ.fastq.gz'
echo""

#make fastq repo + dl fastq data
echo "mk fastq repo + dl fastq reps..."
data_dir=/nobackup/$USER/DISS/data
fastq_dir=/nobackup/$USER/DISS/data/fastq
if [ ! -d $fastq_dir ]
then
    mkdir $fastq_dir
fi
cd ${fastq_dir}
wget -nc $rep_1_URL 
wget -nc $rep_2_URL 
gunzip *.gz
echo "fastq success!" 
echo""

#fastqc
echo "mk fastqc repo + fastqc for both reps..."
fastq_dir=/nobackup/$USER/DISS/data/fastq
fastqc_dir=/nobackup/$USER/DISS/data/fastqc
if [ ! -d $fastqc_dir ]
then
    mkdir $fastqc_dir
fi

for file in ${fastq_dir}/*.fastq; do
    output_file="$fastqc_dir/$(basename "$file" .fastq)_fastqc.zip"
    if [ ! -f "$output_file" ]; then
        fastqc -o "$fastqc_dir" "$file"
    else
        echo "Output file $output_file already exists. Skipping..."
    fi
done
echo "fastqc success!" 
echo""

#trimming
module load trim_galore
