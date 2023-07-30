#!/usr/bin/bash
cd /nobackup/$USER/PRJNA215956/genome
gunzip Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz
more Arabidopsis_thaliana.TAIR10.dna.toplevel.fa
grep ">" Arabidopsis_thaliana.TAIR10.dna.toplevel.fa
