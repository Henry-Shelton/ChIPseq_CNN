Detailed docs can be found in paper/supplementary data [M&M]
3.1	Data Acquisition and Selection
3.2	ChIP-seq Pipeline 
3.3	Peak calling
3.4	EDA, Prediction and Visualisation of TFBMs
3.5	EDA, Prediction and Visualisation of TFBMs
3.6	CNN Data Preprocessing
3.7	CNN Model Architecture
3.8	Evaluation, Loss Function and Peak scoring
3.9	CNN Training
3.10	Hyperparameter Optimisation
3.11	CNN Peak Calling
3.12	Comparative Analysis
4	RESULTS

<br>

# [1] preprocessing data (bioinfo pipeline)

requirements: samtools/bamtools/fastqc/multiqc/bowtie/macs2

Obtain FASTQ + Control raw data from ENCODE
Use src/ scripts to QC/trim/align/sort/index -> output = .bam

## [1.1] MACS2 peakcalling (standard pipeline)

<br>
<br>

# [2] CNN peak calling model

REQUIREMENTS.TXT for module version control/install (virtualenv)

<br>

## [2.1] preprocess data for CNN

aligned BAM
label data
ref seq files

```sh
python preproc.py
```

test/train data outputs

<br>

## [2.2] buildmodel

for hyperparam optimisation + building models + evaluation of models
call
```sh
python buildmodel.py
```
hyperparams = define hyperparams:

<p align="left">
    <img src="pics/defineHP.png">
</p>

definemodel = model architecture:

<p align="left">
    <img src="pics/definemodel.png">
</p>

buildmodel = split data/statistics/generations/train params/output+save learned weights:

<p align="left">
    <img src="pics/buildmodel.png">
</p>


workflow:
```sh
hyperparameters.py -> definemodel.py -> buildmodel.py
```
outputs model + eval stats:

<p align="left">
    <img src="pics/modeloutput_evals.png">
</p>

<br>

## [2.3] using trained model to predict peaks

requires cythonating C files found in setup.py 
calls all buildmodel scripts BUT with learned model

call
```sh
python callpeaks.py
```

<br>

## [2.4] output processing

output peak files + test model metrics

call
```sh
python makeScore.py
python errorCall.py
```
compare to traditional methods (MACS2 etc)

######################################################

<br>
<br>

# [3] final comparisons + motif output

back to shell scripts
deeptools fpr heatmap plots and profiles
correlation metrics

DREAM for motif analysis
Read paper for more info

<br>
