#!/usr/bin/env bash
cd /nobackup/$USER/PRJNA215956
mkdir genome
cd genome
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-56/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
wget ftp://ftp.ensemblgenomes.org/pub/plants/release-56/gtf/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.56.gtf.gz
