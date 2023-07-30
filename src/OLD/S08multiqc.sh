#!/usr/bin/bash
cd /nobackup/$USER/PRJNA215956
mkdir multiqc_report
cd multiqc_report
module load bioinformatics
module load multiqc
export PYTHONIOENCODING=utf-8 #Set the Python encoding to UTF-8
multiqc  --help # to display the help page
multiqc -n multiqc_report_fastqc /nobackup/$USER/PRJNA215956/fastqc/*zip
#firefox multiqc_report_fastqc.html
