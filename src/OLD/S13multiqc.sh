#!/usr/bin/bash
cd /nobackup/$USER/PRJNA215956/multiqc_report
module load multiqc
export PYTHONIOENCODING=utf-8 #Set the Python encoding to UTF-8
multiqc -n multiqc_report_bam_stats /nobackup/$USER/PRJNA215956/work/*sorted_bam_stats.txt
#firefox multiqc_report_bam_stats.html
