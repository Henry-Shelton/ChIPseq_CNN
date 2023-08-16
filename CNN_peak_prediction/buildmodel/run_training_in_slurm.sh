#!/bin/bash

#SBATCH -p shared                              # You select the queue(cluster) here
#SBATCH -c 96                                 # number of CPU cores to allocate, one per thread, up to 128.
#SBATCH --mem=190G                             # memory required, in units of k,M or G, up to 250G.
#SBATCH --gres=tmp:300G                        # $TMPDIR space required on each compute node, up to 400G.
#SBATCH -t 24:00:00                            # time limit in format dd-hh:mm:ss

#SBATCH --job-name=CNN_Training                # Remove the spaces around the equal sign
#SBATCH --output=training%A_%a.out
#SBATCH --error=training%A_%a.err

cd /nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/
source virtualenv/bin/activate

cd /nobackup/kfwc76/DISS/CNN_Peak_Calling_algo/buildmodel/
python buildmodel.py
