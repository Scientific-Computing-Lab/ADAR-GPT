# ADAR-GPT

## Overview
RNA editing is a crucial post-transcriptional mechanism that alters RNA sequences, impacting gene regulation and disease. This repository contains code for predicting adenosine-to-inosine (A-to-I) RNA editing sites using GPT-4o-mini with a continual fine-tuning (CFT) strategy.

We introduce a liver-specific dataset, where ADAR1 is the predominant enzyme, and train models progressively on editing thresholds (1%, 5%, 10%, 15%), improving classification accuracy compared to traditional fine-tuning methods.

### Key Contributions:
   - Liver-Specific RNA Editing Analysis: Avoiding confounding multi-tissue variability.
   - Continual Fine-Tuning (CFT): Training the model step-by-step from low (1%) to high (15%) editing levels.
   - Non-Overlapping Thresholds: Each site assigned a single editing category to improve classification accuracy.
   - Improved Performance: Outperforms GPT-3.5 and static fine-tuning (SFT) models.

## 🧬 Methodology
Our approach focuses on improving RNA editing site prediction using transformer-based models, specifically **GPT-4o-mini**, in a **continual fine-tuning (CFT)** paradigm. This methodology allows the model to progressively learn from lower to higher editing thresholds (1%, 5%, 10%, 15%), refining its understanding of RNA editing patterns.

We trained the model using a **liver-specific dataset** derived from GTEx, ensuring minimal interference from non-relevant ADAR isoforms. The training procedure included:

A) **Data Collection & Preprocessing**
   - Extracting double-stranded RNA (dsRNA) structures from Alu elements.
   - Annotating editing levels based on GTEx liver data.
   - Predicting RNA secondary structures using ViennaRNA(RNAfold) and converting into Vienna format.

   **Data Partitioning**
   - Overlapping Sites: Multiple thresholds assigned per site (e.g., 1-5%, 5-10%).
   - Non-Overlapping Sites: Each site belongs to one distinct threshold to ensure clearer distinctions.

B) **RNA Editing as a Classification Problem**

   - Framing RNA editing site prediction as a binary classification task.
   - The model determines whether a given adenosine is edited (Yes/No) based on its sequence and structure.
   - Training labels are derived from GTEx data, assigning a binary label to each adenosine.

C) **Comparing Fine-Tuning Strategies (SFT vs. CFT)**

   - Static Fine-Tuning (SFT): Training on a single threshold (e.g., only 15% editing).
   - Continual Fine-Tuning (CFT): Gradual training from low (1%) to high (15%) editing levels.
   - CFT enables better adaptation across editing ranges, leading to more robust classification performance.
     
![methodology](Figure/methodology/methodology.png)

## Repository Structure

   
## Getting Started
### Requirments

First, clone this repository. 

You may use the file  `environment.yml` to create anaconda environment (python 3.8) with the required packages.

### Steps to Use the environment.yml File:
#### Create the Environment:
1. Save the `environment.yml` file in your project directory, then run the following command:
   
```
conda env create -f environment.yml
```

2. Activate the Environment:
   
```
conda activate A2IRnaEditing
```

## Data Preparation

### 1. Classification Task

For the classification task, data preparation involves extracting RNA sequences, computing secondary structures, and assigning editing labels for liver tissue.
Classification Data Creation Script: This script generates the dsRNA structure and processes RNA sequences to classify editing sites based on their structural and sequence context.

To run this script, navigate to the Script/data_preparation folder and use the following command:

```
python Classification_Data_Creation_Liver.py [-h] --pair_region PAIR_REGION --output_dir OUTPUT_DIR
                                        --editing_site_plus EDITING_SITE_PLUS
                                        --editing_site_minus EDITING_SITE_MINUS --genome GENOME
```
Outputs:
   - data_for_prepare_classification.csv – Processed classification data

### 2. Data Balancing by Editing Thresholds

Data balancing ensures equal representation of edited and non-edited sites across different editing levels, preventing bias in model training.

#### Overlapping Sites

To generate balanced classification datasets for different editing thresholds, navigate to the Script/data_preparation directory and run the following command:
```
Rscript Division_thresholds_overlapping.R -i <input file(data_for_prepare_classification.csv)> -o < output_dir>
```
This script divides the dataset into overlapping editing levels (1%, 5%, 10%, 15%) and ensures balanced distributions of edited and non-edited sites. The output consists of four files, each corresponding to a different threshold.

#### Non-Overlapping Sites

For non-overlapping classification thresholds, use the following command:
```
Rscript Division_thresholds_non_overlapping.R -i <input file(data_for_prepare_classification.csv)> -o < output_dir>
```
This script allows a site to belong to multiple editing level categories, resulting in four output files similar to the overlapping approach.

### 3.Preparing Data for GPT Fine-Tuning

To prepare the data for GPT fine-tuning, navigate to the Script/data_preparation directory and run the following command:
```
python Model_Input_Preparation_Classification.py <input_csv>
```
This script processes the classification dataset into a structured JSONL format for model training and evaluation.
Outputs:
   - train_<timestamp>.jsonl – Training dataset
   - valid_<timestamp>.jsonl – Validation dataset



## Inference

The inference process differs based on the training methodology used. In CFT (Continual Fine-Tuning), inference is performed iteratively, where each model serves as the basis for fine-tuning the next model. The key distinction is that each inference step is applied to a different model fine-tuned on progressively refined data. This approach allows for continuous improvement in predictions across multiple runs. In contrast, SFT (Single Fine-Tuning) involves training a model directly on a dataset with a specific editing level, making inference a one-step process where the model is applied directly to new data without iterative refinements.

To perform inference, navigate to the Script/inferencing directory and run the following command:
```
 python inferencing.py <input_file> <output_file> <temperature> 
```
Input: The <input_file> is the file created in the Model_Input_Preparation_Classification.py step.

## Baseline Comparisons

We provide scripts to compare ADAR-GPT against two state-of-the-art baselines: EditPredict (CNN) and RNA-FM (foundation model).

### Installing Baseline Models

#### EditPredict

**Installation**:
```bash
git clone https://github.com/wjd198605/EditPredict.git
cd EditPredict
pip install tensorflow==2.11.0 keras scikit-learn pandas numpy
```

**Required files** (included in repository):
- `editPredict_weight_alu.json` – Model architecture
- `editPredict_construction_alu.h5` – Pre-trained weights

#### RNA-FM

**Installation**:
```bash
git clone https://github.com/ml4bio/RNA-FM.git
cd RNA-FM
pip install torch transformers multimolecule
pip install -e .
```
#### Pre-trained vs. Retrained Models

- **Pre-trained**: Model trained on original authors' dataset . Quick to evaluate but may not generalize to your data.
- **Retrained**: Model trained from scratch on your exact dataset. Provides fair comparison. In our experiments: pre-trained EditPredict (F1=0.67) vs retrained (F1=0.78).

#### Both-Strand Scoring

The `--both_strands` flag evaluates both forward sequence and reverse-complement, taking the maximum probability. This can improve recall but may increase false positives.

### 1. EditPredict Baseline

#### Option A: Quick Evaluation (Pre-trained Model)

**Script**: `adar_gpt_vs_editpredict.py`

**Input**:
- `--train_jsonl`: Training JSONL file
- `--valid_jsonl`: Validation JSONL file
- `--editpredict_dir`: Path to EditPredict repository
- `--threshold`: Decision threshold (default: 0.5)
- `--outdir`: Output directory

**Command**:
```bash
python Script/baselines/EditPredict/adar_gpt_vs_editpredict.py \
  --train_jsonl path/to/train.jsonl \
  --valid_jsonl path/to/valid.jsonl \
  --editpredict_dir path/to/EditPredict \
  --threshold 0.5 \
  --outdir editpredict_baseline_out
```

**Output**:
- `editpredict_probs_valid.csv` – Probabilities for each site
- `editpredict_metrics_valid.json` – Performance metrics

#### Option B: Enhanced Evaluation

**Script**: `adar_gpt_vs_editpredict_plus.py`

**Input**: Same as Option A, plus:
- `--both_strands`: (Optional) Evaluate both strand orientations

**Command**:
```bash
python Script/baselines/EditPredict/adar_gpt_vs_editpredict_plus.py \
  --train_jsonl path/to/train.jsonl \
  --valid_jsonl path/to/valid.jsonl \
  --editpredict_dir path/to/EditPredict \
  --both_strands \
  --outdir editpredict_plus_out
```

**Output**:
- `probs.csv` – Raw probabilities
- `metrics@0.5.json` – Metrics at threshold 0.5
- `metrics@bestF1.json` – Best F1 score with optimal threshold
- `threshold_sweep.csv` – Metrics across 101 thresholds
- `roc_curve.csv` – ROC curve data
- `pr_curve.csv` – Precision-Recall curve data

#### Option C: Retrain on Your Data

**Script**: `retrain_editpredict_pipeline.py`

**Input**:
- `--train_jsonl`: Training JSONL file
- `--valid_jsonl`: Validation JSONL file
- `--editpredict_dir`: Path to EditPredict repository
- `--outdir`: Output directory
- `--input_len`: Window length (default: 201)
- `--epochs`: Training epochs (default: 10)
- `--batch_size`: Batch size (default: 128)
- `--evaluate_with_plus`: (Optional) Run enhanced evaluation after training

**Command**:
```bash
python Script/baselines/EditPredict/retrain_editpredict_pipeline.py \
  --train_jsonl path/to/train.jsonl \
  --valid_jsonl path/to/valid.jsonl \
  --editpredict_dir path/to/EditPredict \
  --outdir editpredict_retrained \
  --epochs 10 \
  --evaluate_with_plus
```

**Output**:
- `data/` – Converted training data
- `model_ep_retrained/` – Retrained model (JSON + H5 files)
- `evaluation/` – (If `--evaluate_with_plus` used) Comprehensive evaluation results

### 2. RNA-FM Baseline

**Script**: `rnafm_finetune_adar.py`

**Input**:
- `--train_jsonl`: Training JSONL file
- `--valid_jsonl`: Validation JSONL file
- `--outdir`: Output directory
- `--model_id`: HuggingFace model ID (default: multimolecule/rnafm)
- `--window_len`: Sequence window length (default: 201)
- `--epochs`: Training epochs (default: 5)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 3e-5)
- `--both_strands`: (Optional) Evaluate both strand orientations

**Command**:
```bash
python Script/baselines/RNA-FM/rnafm_finetune_adar.py \
  --train_jsonl path/to/train.jsonl \
  --valid_jsonl path/to/valid.jsonl \
  --outdir rnafm_finetuned \
  --epochs 5 \
  --batch_size 32 \
  --both_strands
```

**Output**:
- `rnafm_finetuned_model/` – Fine-tuned model checkpoint
- `probs.csv` – Prediction probabilities
- `metrics@0.5.json` – Fixed threshold metrics
- `metrics@bestF1.json` – Best F1 performance
- `threshold_sweep.csv` – Full threshold analysis
- `roc_curve.csv`, `pr_curve.csv` – Performance curves

## Performance

### Baseline Comparison (All Thresholds)

| Threshold | Model | Best-F1 | t* | Accuracy | Recall | Specificity |
|-----------|-------|---------|-----|----------|--------|-------------|
| **1%** | EditPredict Pre-trained (both) | 0.6730 | 0.90 | 0.5211 | 0.9816 | 0.0571 |
| | EditPredict Retrained (single) | 0.6956 | 0.16 | 0.6172 | 0.8714 | 0.3611 |
| | EditPredict Retrained (both) | 0.6750 | 0.07 | 0.5211 | 0.9908 | 0.0478 |
| | RNA-FM Fine-tuned (both) | 0.7054 | 0.74 | 0.5942 | 0.9678 | 0.2176 |
| | **ADAR-GPT CFT (1%)** | **0.7956** | – | **0.7428** | **0.7248** | **0.7832** |
| **5%** | EditPredict Pre-trained (both) | 0.6791 | 0.99 | 0.5503 | 0.9582 | 0.1481 |
| | EditPredict Retrained (single) | 0.7145 | 0.04 | 0.6480 | 0.8870 | 0.4122 |
| | EditPredict Retrained (both) | 0.6849 | 0.10 | 0.5657 | 0.9505 | 0.1863 |
| | RNA-FM Fine-tuned (both) | 0.7149 | 0.91 | 0.6610 | 0.8560 | 0.4687 |
| | **ADAR-GPT CFT (5%)** | **0.7994** | – | **0.7655** | **0.7946** | **0.7241** |
| **10%** | EditPredict Pre-trained (both) | 0.6751 | 0.96 | 0.5288 | 0.9800 | 0.0783 |
| | EditPredict Retrained (single) | 0.7400 | 0.06 | 0.6749 | 0.9262 | 0.4240 |
| | EditPredict Retrained (both) | 0.6922 | 0.16 | 0.5803 | 0.9446 | 0.2166 |
| | RNA-FM Fine-tuned (both) | 0.7207 | 0.91 | 0.6849 | 0.8138 | 0.5561 |
| | **ADAR-GPT CFT (10%)** | **0.7751** | – | **0.7753** | **0.7444** | **0.8089** |
| **15%** | EditPredict Pre-trained (both) | 0.6730 | 0.98 | 0.5273 | 0.9784 | 0.0810 |
| | EditPredict Retrained (single) | 0.7751 | 0.18 | 0.7448 | 0.8841 | 0.6070 |
| | EditPredict Retrained (both) | 0.7251 | 0.70 | 0.6749 | 0.8624 | 0.4893 |
| | RNA-FM Fine-tuned (both) | 0.7542 | 0.97 | 0.7164 | 0.8748 | 0.5596 |
| | **ADAR-GPT CFT (15%)** | **0.7735** | – | **0.7872** | **0.7358** | **0.8373** |
| | ADAR-GPT SFT (15%) | 0.6871 | – | 0.7092 | 0.6468 | 0.7699 |

