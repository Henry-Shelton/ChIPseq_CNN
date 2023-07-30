rm(list=ls())
library(openxlsx)
username<-Sys.getenv("USER")
work_dir<-"/nobackup/prsd58/PRJNA215956/result"
work_dir<-sub("prsd58", username, work_dir)
setwd(work_dir)
sampleNames<-c("BCF2")
for ( i in 1:length(sampleNames) ) {
  sampleName<-sampleNames[i]
  png(filename =paste0(sampleName, "_allele_frequency_plot.png"),
      width = 1320, height = 720, units = "px", pointsize = 12,
      bg = "white",  res = 96)
  par(mfrow = c(2, 3)) 
  inputfile<-paste0(sampleNames[i], "_strelka2.pass.ann.xlsx")
  result<-read.xlsx(xlsxFile=inputfile)
  result<-result[, c("CHROM", "POS", "Allele.Frequency")]
  result<-result[!duplicated(result),]
  nrow(result)
  result<-split(result, result$CHROM)
  for (j in 1:length(result)) {
    af<-result[[j]]
    chr<-names(result)[j]
    o<-order(af$POS)
    af<-af[o,]
    plot(af$POS, af$Allele.Frequency, xlab="POS", 
         ylab="Allele Frequency", ylim=c(0, 1), pch=16, col="red",
         main=paste0(sampleName, "_chr", chr))
    abline(h=0.85, col="blue")
    grid()
    }
  dev.off()
}
