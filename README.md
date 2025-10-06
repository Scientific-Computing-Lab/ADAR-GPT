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



