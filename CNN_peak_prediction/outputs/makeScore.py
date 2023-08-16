import csv
import subprocess as sp
import argparse
import pandas as pd
import math
import subprocess as sp

input_bed = "AKAP8L_Hep2G.bed"
output_csv = "/nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/outputs/output_scores.csv"

col = ['chr','start','end','name','sigVal','pVal','maxpVal','maxRead']

bed = pd.read_csv(input_bed, sep='\t', names=col)

score = []
score_max = []
for i in range(len(bed)):
    pval = bed['pVal'][i]
    max_pval = bed['maxpVal'][i]
    sig_val = bed['sigVal'][i]

    # Check if the p-value and max p-value are positive before calculating the score
    if pval > 0 and max_pval > 0:
        score.append(100 * -math.log10(pval) * sig_val)
        score_max.append(100 * -math.log10(max_pval) * sig_val)
    else:
        score.append(0)
        score_max.append(0)

bed = bed.assign(score=score)
bed = bed.assign(score_max=score_max)

print(bed)

bed[['chr','start','end','name','sigVal','pVal','maxpVal','maxRead','score','score_max']].to_csv(output_csv,sep=',',header=True, index=False)

