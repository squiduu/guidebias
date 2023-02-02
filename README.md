# GuiDebias
Official scripts for "Compensatory Debiasing for Gender Imbalances in Language Models."

## Overview
<p align="center">
<img src="./figs.png">
</p>

## Installation
This repository is available in Ubuntu 18.04.5 LTS, and it is not tested in other OS.
```
git clone https://github.com/squiduu/guidebias.git
cd guidebias

conda create -n guidebias python=3.7.10
conda activate guidebias

pip install -r requirements.txt
```

## Bias mitigation
Fine-tune a pre-trained BERT to debias.
```
cd guidebias
mkdir ./out/
sh run_finetune.sh
```
Then, a debiased BERT model will be saved in `./out/`.

## Download dataset
Download StereoSet test set from [here](https://github.com/McGill-NLP/bias-bench/blob/main/data/stereoset/test.json).
```
mkdir ../stereoset/data/
cd ../stereoset/data/
```
Put the `test.json` in `../stereoset/data/`.

## Evaluation
All of the evaluation scripts are followed [Bias Bench](https://github.com/McGill-NLP/bias-bench/blob/main/data/stereoset/test.json). However, some minor modifications were made to suit our experimental environment.
### For original BERT
**SEAT**
```
cd ../seat/
mkdir ./out/
sh run_seat_original.sh
```
**StereoSet**
```
cd ../stereoset/
mkdir ./out/
sh run_stereoset_original.sh
sh evaluate_original.sh
```
**CrowS-Pairs**
```
cd ../crows_pairs/
mkdir ./out/
sh run_crows_pairs_original.sh
```
**GLUE**
```
cd ../glue/
mkdir ./out/
sh run_glue_original.sh
```

### For debiased BERT
**SEAT**
```
cd ../seat/
sh run_seat_debiased.sh
```
**StereoSet**
```
cd ../stereoset/
rm -rf ./out/results/
sh run_stereoset_debiased.sh
sh evaluate_debiased.sh
```
**CrowS-Pairs**
```
cd ../crows_pairs/
sh run_crows_pairs_debiased.sh
```
**GLUE**
```
cd ../glue/
sh run_glue_debiased.sh
```

## Results
**SEAT**
|Model|SEAT-6|SEAT-6b|SEAT-7|SEAT-7b|SEAT-8|SEAT-8b|Avg.|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BERT|0.931|0.090|-0.124|0.937|0.783|0.858|0.620|
|CDA|0.846|0.186|-0.278|1.342|0.831|0.849|0.722|
|Dropout|1.136|0.317|0.138|1.179|0.879|0.939|0.765|
|Sent-Debias|0.350|-0.298|-0.626|0.458|0.413|0.462|0.434|
|INLP|0.317|-0.354|-0.258|0.105|0.187|-0.004|0.204|
|Context-Debias|0.409|0.159|-0.222|0.848|0.537|0.176|0.392|
|Auto-Debias|0.344|0.016|0.173|1.123|0.734|0.783|0.529|
|Ours|-0.023|-0.249|-0.405|0.144|-0.353|-0.001|**0.196**|

**StereoSet**
|Model|LMS|SS|ICAT|
|:---|:---:|:---:|:---:|
|BERT|84.17|60.28|66.86|
|CDA|83.08|59.61|67.11|
|Dropout|83.04|60.66|65.34|
|Sent-Debias|84.20|59.37|68.42|
|INLP|80.63|57.25|68.94|
|Context-Debias|85.34|59.21|69.62|
|Auto-Debias|74.09|53.11|69.48|
|Ours|83.83|55.36|**74.84**|

**GLUE**
|Model|CoLA|MNLI|MRPC|QNLI|QQP|RTE|SST2|STSB|WNLI|Avg.|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|BERT|55.64|84.12|82.19|91.31|89.23|61.73|92.32|87.75|36.15|75.60|
|CDA|55.31|84.56|82.76|91.16|90.18|65.46|92.54|88.03|32.86|75.87|
|Dropout|50.90|84.37|80.64|91.20|89.94|63.18|92.58|87.42|39.91|75.57|
|Sent-Debias|48.55|84.26|81.86|91.43|90.78|61.37|92.35|87.74|34.74|74.79|
|INLP|55.91|84.09|84.10|91.17|89.15|62.22|92.39|87.83|34.74|75.73|
|Context-Debias|53.91|84.28|82.98|91.43|89.18|61.48|92.24|87.00|36.15|75.41|
|Auto-Debias|55.89|84.25|84.20|91.57|89.21|62.58|92.51|87.68|39.44|76.37|
|Ours|56.15|84.16|86.17|91.26|89.19|62.34|92.39|87.78|39.44|**76.54**|

**CrowS-Pairs**
|Model|SS|AntiSS|Avg.|
|:---|:---:|:---:|:---:|
|BERT|57.86|56.31|7.09|
|CDA|54.09|60.19|7.14|
|Dropout|57.23|55.34|6.29|
|Sent-Debias|37.74|74.76|18.51|
|INLP|42.77|63.11|10.17|
|Context-Debias|61.01|51.46|6.24|
|Auto-Debias|48.43|59.22|5.40|
|Ours|55.35|54.37|**4.86**|
