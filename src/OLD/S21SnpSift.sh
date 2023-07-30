#!/bin/bash
#SBATCH -p shared # You select the queue(cluster) here
#SBATCH -c 1                # number of CPU cores required, one per thread, up to 128.
#SBATCH --mem=2G            # memory required, up to 250G.
#SBATCH --gres=tmp:3G       # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 01:00:00     # time limit in format dd-hh:mm:ss
#SBATCH --job-name=S21SnpSift # This name will let you follow your job
#SBATCH --output=../log/S21SnpSift%A_%a.out
#SBATCH --error=../log/S21SnpSift%A_%a.err
#SBATCH --array=1
work_dir=/nobackup/$USER/PRJNA215956/work
sample_names=("" "BCF2")
module load bioinformatics
module load snpeff/5.0
sample=${sample_names[$SLURM_ARRAY_TASK_ID]}
cd ${work_dir}/${sample}_strelka/results/variants/snpEff_snvs
cat somatic.snvs.pass.ann.vcf | vcfEffOnePerLine.pl > somatic.snvs.pass.ann.oneperline.vcf
java -jar /nobackup/dbl0hpc/apps/snpEff/SnpSift.jar \
      extractFields somatic.snvs.pass.ann.oneperline.vcf \
      CHROM  POS   REF  FILTER  ANN[*].ALLELE ANN[*].EFFECT ANN[*].IMPACT ANN[*].GENE ANN[*].GENEID  ANN[*].FEATURE \
      ANN[*].FEATUREID ANN[*].BIOTYPE ANN[*].HGVS_C ANN[*].HGVS_P GEN[TUMOR].DP GEN[TUMOR].AU[0] \
      GEN[TUMOR].CU[0] GEN[TUMOR].GU[0] GEN[TUMOR].TU[0] >  \
      somatic.snvs.pass.ann.oneperline.txt

cd ${work_dir}/${sample}_strelka/results/variants/snpEff_indels
cat somatic.indels.pass.ann.vcf | vcfEffOnePerLine.pl > somatic.indels.pass.ann.oneperline.vcf
java -jar /nobackup/dbl0hpc/apps/snpEff/SnpSift.jar \
      extractFields somatic.indels.pass.ann.oneperline.vcf \
      CHROM  POS   REF  FILTER  ANN[*].ALLELE ANN[*].EFFECT ANN[*].IMPACT ANN[*].GENE ANN[*].GENEID  ANN[*].FEATURE \
      ANN[*].FEATUREID ANN[*].BIOTYPE ANN[*].HGVS_C ANN[*].HGVS_P GEN[TUMOR].DP GEN[TUMOR].TAR[0] GEN[TUMOR].TIR[0] >  \
      somatic.indels.pass.ann.oneperline.txt
