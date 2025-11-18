# ADAR-GPT

## Overview
RNA editing is a crucial post-transcriptional mechanism that alters RNA sequences, impacting gene regulation and disease. This repository contains code for predicting adenosine-to-inosine (A-to-I) RNA editing sites using GPT-4o-mini with a continual fine-tuning (CFT) strategy.

We introduce a liver-specific dataset derived from GTEx (n=131 samples), where ADAR1 is the predominant enzyme, enabling controlled analysis of ADAR1-mediated editing. Our two-stage continual fine-tuning approach first trains on curriculum data from all editing thresholds (1%→5%→10%→15%) followed by refinement on high-confidence 15% sites, achieving superior performance compared to static fine-tuning and established baselines.

### Key Contributions:
   - Liver-Specific RNA Editing Analysis: Avoiding confounding multi-tissue variability.
   - Continual Fine-Tuning (CFT): Training the model step-by-step from low (1%) to high (15%) editing levels.
   - Non-Overlapping Thresholds: Each site assigned a single editing category to improve classification accuracy.
   - Improved Performance: F1=76.3%, accuracy=75.9% at 15% threshold, outperforming CNN and foundation model baselines

## 🧬 Methodology
Our approach improves RNA editing site prediction using **GPT-4o-mini** with a **two-stage continual fine-tuning (CFT)** paradigm. This methodology enables progressive learning from curriculum data across all thresholds before specializing on high-confidence editing sites.

We trained the model using a **liver-specific dataset** derived from GTEx, ensuring minimal interference from non-ADAR1 isoforms. The training procedure included:

A) **Data Collection & Preprocessing**
   - Identifying oppositely oriented Alu element pairs within UTRs to establish genomic context
   - Quantifying editing levels for each adenosine from GTEx liver RNA-seq data (n=131 samples)
   - Extracting 201-nucleotide windows centered on candidate adenosines (100 upstream + target + 100 downstream)
   - Filtering for sites with >100 read coverage to ensure reliability

   **Data Partitioning: Non-Overlapping Threshold Groups**
   Each adenosine site was assigned to exactly one threshold group based on its editing level:
   - **1% group**: 1% ≤ editing < 5% (positives) vs. editing < 1% (negatives)
   - **5% group**: 5% ≤ editing < 10% (positives) vs. editing < 5% (negatives)  
   - **10% group**: 10% ≤ editing < 15% (positives) vs. editing < 10% (negatives)
   - **15% group**: editing ≥ 15% (positives) vs. editing < 15% (negatives)

B) **RNA Editing as a Classification Problem**

   - Framing RNA editing site prediction as a binary classification task.
   - The model determines whether a given adenosine is edited (Yes/No) based on its sequence.
   - Training labels are derived from GTEx data, assigning a binary label to each adenosine.

C) **Comparing Fine-Tuning Strategies (SFT vs. CFT)**

   - Static Fine-Tuning (SFT): Training on a single threshold (e.g., only 15% editing).
   - Continual Fine-Tuning (CFT): Gradual training from low (1%) to high (15%) editing levels.
   - CFT enables better adaptation across editing ranges, leading to more robust classification performance.
     
![methodology](Figure/methodolgy.png)

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
     

### 2. Non-Overlapping Threshold Groups

Create balanced datasets where each adenosine site belongs to exactly one threshold group - 
navigate to the Script/data_preparation directory and run the following command:

```
Rscript build_equal_groups.R \
  --input_csv data_for_prepare_classification.csv \
  --output_dir Output directory for all CSVs
```
This script partitions adenosines into four mutually exclusive groups:
- **1% group**: 1% ≤ editing < 5% (positives) vs. editing < 1% (negatives)
- **5% group**: 5% ≤ editing < 10% (positives) vs. editing < 5% (negatives)  
- **10% group**: 10% ≤ editing < 15% (positives) vs. editing < 10% (negatives)
- **15% group**: editing ≥ 15% (positives) vs. editing < 15% (negatives)

**Output**: Four balanced CSV files, each containing equal numbers of positive and negative examples for the respective threshold.

### 3.Preparing Data for GPT Fine-Tuning

To prepare the data for GPT fine-tuning, navigate to the Script/data_preparation directory and run the following command:

```
python csv_to_jsonl_GPT.py \
  --input_csv  \
  --output_jsonl
```

This creates structured conversations with:
- **System prompt**: Task instruction for RNA editing prediction
- **User input**: Sequence context with structure information
- **Assistant response**: Binary classification (Yes/No for editing)

### 4. Normalize Sequence Windows

Trim sequences to standardized 201-nucleotide windows (100 upstream + target A + 100 downstream):
```bash
python python trim_jsonl.py <input.jsonl> <output.jsonl>
```

This step:
- Extracts exactly 100 bases upstream and downstream of target adenosine
- Pads with 'N' if insufficient flanking sequence
- Removes structure information to focus on sequence-only input

**Note**: Example files showing the expected input/output formats are provided in the `input_file/` directory. The provided files contain sample data (100-1,000 examples) for format demonstration. Use these as templates for your own data preparation.

**File Naming Convention:**
- `X_YP_*.jsonl`: Contains sites with X% ≤ editing < Y% (positives) vs. editing < X% (negatives)
- `15P_*.jsonl`: Contains sites with editing ≥ 15% (positives) vs. editing < 15% (negatives)
- `*_201L_*.jsonl`: Trimmed to 201-nucleotide windows, structure information removed
- `CFT_*`: Combined curriculum dataset (all thresholds) for continual fine-tuning
- `SFT_*`: Single threshold dataset for static fine-tuning

## Inference

The inference process evaluates your trained ADAR-GPT model on new RNA sequences and provides probability scores for editing prediction.

### Prerequisites

Before running inference, ensure you have:
1. A fine-tuned ADAR-GPT model deployed on Azure OpenAI
2. Azure OpenAI credentials configured (endpoint, API key, deployment name)
3. Validation data in the correct JSONL format (from step 4 above)

### Setting Up Azure Credentials

Configure your Azure OpenAI environment using the provided script:
```bash
# Navigate to the inference directory
cd Script/inferencing

# Set up your Azure credentials (edit the script with your actual values)
source set_azure_credentials.sh
```

The credentials script sets the following environment variables:
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI resource endpoint
- `AZURE_OPENAI_DEPLOYMENT` - Name of your deployed ADAR-GPT model  
- `AZURE_OPENAI_API_VERSION` - Azure OpenAI API version
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI access key

### Running Inference

To perform inference on your validation dataset, use the evaluation script:
```bash
python evaluation_script.py \
  --dataset path/to/your_validation_trimmed.jsonl \
  --log-file logs/model_predictions.jsonl \
  --metrics-file logs/performance_metrics.json \
  --positive-label "yes" \
  --threshold 0.5 \
  --rpm 10 \
  --progress
```

**Parameter Explanation:**
- `--dataset` - Path to your validation JSONL file (output from trim_jsonl.py)
- `--log-file` - Where to save detailed per-example predictions  
- `--metrics-file` - Where to save aggregate performance metrics
- `--positive-label` - Label to treat as positive class ("yes" for editing sites)
- `--threshold` - Decision threshold for binary classification (0.5 = 50%)
- `--rpm` - Requests per minute (controls API rate limiting)
- `--progress` - Show progress updates during inference

**Our Results:**
The complete validation results from our experiments are organized in the `logs/` directory:
- Model predictions: `logs/model_outputs/model_outputs_15_CFT_NoStructureonly100_FT15.jsonl` (CFT) and `model_outputs_15_SFT_NoStructureonly100.jsonl` (SFT)
- Performance metrics: `logs/metrics/` (JSON files reporting accuracy, F1, AUROC, AUPRC, etc.)

## Independent Test Set Evaluation

To verify generalization to completely novel Alu sequences, we evaluated ADAR-GPT on 1,288 independent test examples from Sun et al. (Nature Biotechnology 2025) with no overlap to our training set.

**Our Results:**
- Model predictions: `logs/model_outputs/model_outputs_independent_alu`
- Performance metrics: `logs/metrics/metrics_independent_alu`

## Curriculum vs. Random Order Control

Control experiment: same 20,600 examples in random order + 15% refinement stage.

**Our Results:**
- `logs/model_outputs/model_outputs_shuff` (if exists)
- `logs/metrics/metrics_shuff` (if exists)
- 

## Biological Validation

Test the model's ability to capture known ADAR sequence preferences using controlled mutations.

### Upstream G Avoidance Test

Test ADAR's preference to avoid guanosine (G) immediately upstream of the editing site:
```bash
cd Script/biological_validation

python test_upstream_g_avoidance.py \
  input_validation.jsonl \
  original_no_upstream_G.jsonl \
  mutated_with_upstream_G.jsonl
```

This script:
- Selects sites that naturally lack G at position -1
- Creates original set (baseline) and mutated set (G forced at -1)
- Expected result: Dramatic suppression of editing predictions in mutated set

### Position Specificity Control

Test model specificity using downstream position +5 (should show no effect):
```bash
python test_position_control.py \
  input_validation.jsonl \
  original_no_downstream_G.jsonl \
  mutated_with_downstream_G.jsonl \
  --offset 5
```

This script:
- Tests G introduction at position +5 after central adenosine
- Expected result: No significant performance difference between original vs. mutated
- Confirms model's sensitivity is position-specific, not general G-content response

### Running Validation
```
# Test both conditions and run inference
python test_upstream_g_avoidance.py validation.jsonl upstream_orig.jsonl upstream_mut.jsonl
python test_position_control.py validation.jsonl downstream_orig.jsonl downstream_mut.jsonl
```

**Our Validation Results:**
Complete results from our biological validation experiments are available in `logs/`:
- Upstream G avoidance test: 
  - Original: `model_outputs/model_outputs_without_G_1_up` and `metrics/metrics_without_G_1_up`
  - Mutated (G at -1): `model_outputs/model_outputs_mutate_G_1_up` and `metrics/metrics_mutate_G_1_up`
- Position +5 control test:
  - Original: `model_outputs/model_outputs_without_G_5_down` and `metrics/metrics_without_G_5_down`
  - Mutated (G at +5): `model_outputs/model_outputs_mutate_G_5_down` and `metrics/metrics_mutate_G_5_down`
- G-only baseline classifier: `metrics/metrics_baseline_G.jsonl`

These files contain the actual predictions and metrics reported in our manuscript (Table 3).

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

#### Option B: Retrain on Your Data

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
```

**Output**:
- `rnafm_finetuned_model/` – Fine-tuned model checkpoint
- `probs.csv` – Prediction probabilities
- `metrics@0.5.json` – Fixed threshold metrics
- `metrics@bestF1.json` – Best F1 performance
- `threshold_sweep.csv` – Full threshold analysis
- `roc_curve.csv`, `pr_curve.csv` – Performance curves

**Our Baseline Results:**
The complete baseline comparison results from our manuscript are available in:
- Prediction probabilities: `logs/model_outputs/baseline_probs/`
  - `editpredict_pretrained_probs.csv` (Pre-trained EditPredict)
  - `editpredict_finetuned_probs.csv` (Retrained EditPredict)
  - `rnafm_finetuned_probs.csv` (Fine-tuned RNA-FM)
- Performance metrics: `logs/metrics/`
  - `metrics_EditPredict_pretrain.json`
  - `metrics_EditPredict_retrain.json`
  - `metrics_RNA-FM.json`

These correspond to the results reported in Table 1 of our manuscript.

## Performance


### Performance Comparison (15% Validation Set)
**Note**: All metrics reported at fixed decision threshold of 0.5.

| Model | Training | F1 | Accuracy | Recall | Specificity | Precision | AUROC | AUPRC |
|-------|----------|----|---------|---------|-----------| ---------|-------|-------|
| **ADAR-GPT (CFT)** | Curriculum + 15% FT | **0.763** | **0.759** | 0.769 | 0.750 | 0.757 | **0.841** | **0.801** |
| ADAR-GPT (SFT) | Static 15% | 0.742 | 0.746 | 0.726 | **0.767** | **0.760** | 0.830 | 0.793 |
| EditPredict (Retrained) | Static 15% | 0.718 | 0.723 | 0.701 | 0.745 | 0.736 | 0.801 | 0.771 |
| RNA-FM (Fine-tuned) | Static 15% | 0.709 | 0.590 | 0.989 | 0.185 | 0.552 | 0.713 | 0.681 |
| EditPredict (Pre-trained) | Static 15% | 0.673 | 0.511 | 1.000 | 0.014 | 0.507 | 0.617 | 0.593 |

### Performance Curves (ROC and Precision-Recall)

![ROC Curves](Figure/liver_roc_curves.png)
*Figure: ROC curves comparing ADAR-GPT variants against baseline models on 15% liver validation set.*

![PR Curves](Figure/liver_pr_curves.png) 
*Figure: Precision-Recall curves for the same comparison.*


Generate ROC and Precision-Recall curves comparing ADAR-GPT variants against baseline models:
```bash
cd Figure
python generate_performance_curves.py
```

**Prerequisites:**
- Model prediction outputs in `logs/model_outputs/` directory
- Baseline prediction CSVs in `logs/model_outputs/baseline_probs/`

**Outputs:**
- `liver_roc_curves.png` - ROC curves for all models
- `liver_pr_curves.png` - Precision-Recall curves for all models

