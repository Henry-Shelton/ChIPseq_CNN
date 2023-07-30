#!/usr/bin/bash
module load bioinformatics
module load fastqc
which fastqc #Display the full filename
fastqc -h #Display the help file
cd /nobackup/$USER/PRJNA215956/src
gedit S06fastqc.sh
sbatch S06fastqc.sh #suubmit the job
squeue -u $USER #check job status
cd /nobackup/$USER/PRJNA215956/fastqc
ls -l
firefox SRR955635_1_fastqc.html #open the html file with firefox
