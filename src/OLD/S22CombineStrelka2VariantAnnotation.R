library(openxlsx)
username<-Sys.getenv("USER")
work_dir<-"/nobackup/prsd58/PRJNA215956/work"
work_dir<-sub("prsd58", username, work_dir)
result_dir<-"/nobackup/prsd58/PRJNA215956/result/"
result_dir<-sub("prsd58", username, result_dir)
if (!file.exists(result_dir)){
    dir.create(result_dir)
}
setwd(work_dir)
annotation<-read.delim(file="../genome/mart_export_Athaliana_gene_annotation.txt", stringsAsFactors=FALSE)
annotation<-annotation[,c("Gene.stable.ID", "Gene.description")]
sampleNames<-c("BCF2")
for (sampleName in sampleNames) {
  fileName<-paste0(sampleName, "_strelka/results/variants/snpEff_snvs/somatic.snvs.pass.ann.oneperline.txt")
  snvs<-read.delim(fileName, stringsAsFactors=FALSE, colClasses = c("character"))
  colnames(snvs)<-sub("ANN....", "", colnames(snvs), fixed=TRUE)
  idx<-grep("DP", colnames(snvs))
  colnames(snvs)[idx]<-"Read Depth"
  snvs$POS<-as.numeric(snvs$POS)
  snvs$AltCounts<-NA
  colnames(snvs)<-sub("GEN.TUMOR..", "", colnames(snvs), fixed=TRUE)
  colnames(snvs)<-sub(".0.", "", colnames(snvs), fixed=TRUE)
  snvs$"AU"<-as.numeric(snvs$"AU")
  snvs$"CU"<-as.numeric(snvs$"CU")
  snvs$"GU"<-as.numeric(snvs$"GU")
  snvs$"TU"<-as.numeric(snvs$"TU")
  snvs$AltCounts[snvs$ALLELE=="A"]<-snvs$"AU"[snvs$ALLELE=="A"]
  snvs$AltCounts[snvs$ALLELE=="C"]<-snvs$"CU"[snvs$ALLELE=="C"]
  snvs$AltCounts[snvs$ALLELE=="G"]<-snvs$"GU"[snvs$ALLELE=="G"]
  snvs$AltCounts[snvs$ALLELE=="T"]<-snvs$"TU"[snvs$ALLELE=="T"]
  snvs$"Read Depth"<-as.numeric(snvs$"Read Depth")
  snvs$"Allele Frequency"<-snvs$AltCounts/(snvs$AU+snvs$CU+snvs$GU+snvs$TU)
  snvs$sampleName<-sampleName
  snvs$mutationID<-paste(snvs$CHROM, snvs$POS, snvs$REF, snvs$ALLELE, sep="_")

  fileName<-paste0(sampleName, "_strelka/results/variants/snpEff_indels/somatic.indels.pass.ann.oneperline.txt")
  indels<-read.delim(fileName, stringsAsFactors=FALSE, colClasses = c("character"))
  colnames(indels)<-sub("ANN....", "", colnames(indels), fixed=TRUE)
  idx<-grep("DP", colnames(indels))
  colnames(indels)[idx]<-"Read Depth"
  indels$POS<-as.numeric(indels$POS)
  indels$AltCounts<-NA
  colnames(indels)<-sub("GEN.TUMOR..", "", colnames(indels), fixed=TRUE)
  colnames(indels)<-sub(".0.", "", colnames(indels), fixed=TRUE)
  indels$"TAR"<-as.numeric(indels$"TAR")
  indels$"TIR"<-as.numeric(indels$"TIR")
  indels$AltCounts<-indels$"TIR"
  indels$"Read Depth"<-as.numeric(indels$"Read Depth")
  indels$"Allele Frequency"<-indels$AltCounts/(indels$TAR+indels$TIR)
  indels$sampleName<-sampleName
  indels$mutationID<-paste(indels$CHROM, indels$POS, indels$REF, indels$ALLELE, sep="_")
  indels<-indels[, colnames(indels) %in% colnames(snvs)]
  snvs<-snvs[, colnames(snvs) %in% colnames(indels)]
  result<-rbind(snvs, indels)
  result<-merge(annotation, result, by.x="Gene.stable.ID", by.y="GENEID")
  write.xlsx(result, paste0(result_dir, sampleName, "_strelka2.pass.ann.xlsx"), rownames=FALSE, overwrite=TRUE)
}
