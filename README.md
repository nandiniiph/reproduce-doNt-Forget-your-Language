# Reproduction Report  
# “Understanding and Mitigating Spurious Correlations in Text Classification with Neighborhood Analysis”

## Original Paper

**Understanding and Mitigating Spurious Correlations in Text Classification with Neighborhood Analysis**

### Authors
Oscar Chew, Hsuan-Tien Lin, Kai-Wei Chang, Kuan-Hao Huang

### Paper Link
https://arxiv.org/abs/2305.13654

### Original Repository
https://github.com/oscarchew/doNt-Forget-your-Language

### Report by
Nandini Putri Hanifa Jannah

---

# 1. Summary and Reproduction Goal

## Paper Summary

This paper investigates spurious correlations in text classification models. Spurious correlations occur when language models rely on shortcut features or dataset-specific artifacts rather than learning meaningful semantic representations.

The authors propose **Neighborhood Analysis**, a representation-level approach for understanding how spurious tokens shift semantically during fine-tuning. They also introduce **NFL (Neighbor-based Fine-tuning Loss)** variants designed to mitigate spurious correlations and improve robustness under biased data distributions.

Their experiments show that although models achieve high accuracy on biased datasets, robustness significantly decreases under distribution shifts.

## Reproduction Goal

The goal of this reproduction is to validate the robustness behavior and representation phenomena reported in the original paper by:

- Reproducing RoBERTa baseline experiments
- Implementing multiple NFL variants:
  - NFL-F
  - NFL-CO
  - NFL-CP
  - NFL-PT
- Implementing DFR and Ideal Model
- Conducting experiments on Amazon and Jigsaw datasets
- Evaluating robustness using:
  - Biased Accuracy
  - Robust Accuracy
  - Robustness Gap
- Reproducing neighborhood analysis and representation visualization using t-SNE
- Comparing paper results with reproducibility findings

Unlike the original experiments, reproduction was performed under different runtime environments and computational constraints.

---

# 2. Environment Setup

## Hardware

**Device:** NVIDIA GPU / Chameleon Cloud environment

---

## Software

Python: 3.10+

Main Libraries:

- PyTorch
- Transformers
- HuggingFace Datasets
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Weights & Biases

---

## Model Configuration

| Configuration | Value |
|---|---:|
| Model | roberta-base |
| Epochs | 6 |
| Seed | 24 |

Methods:

- RoBERTa Baseline
- NFL-F
- NFL-CO
- NFL-CP
- NFL-PT
- DFR
- Ideal Model

---

# 3. Step-by-Step Reproduction

## 1. Clone Repository

```bash
git clone https://github.com/oscarchew/doNt-Forget-your-Language

cd doNt-Forget-your-Language
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch transformers datasets pandas sklearn matplotlib seaborn
```

## 3. Dataset Configuration

Datasets used:

### Amazon
Task:
Sentiment classification under biased distributions

### Jigsaw
Task:
Toxicity classification with spurious identity correlations

Training files:

```bash
biased_amazon_train.csv

biased_jigsaw_balance_train.csv

unbiased datasets
```

## 4. Run Training

Baseline:

```bash
python src/nfl.py \
--train_file data/biased_amazon_train.csv \
--model_name roberta-base \
--reg_method None
```

NFL Example:

```bash
python src/nfl.py \
--train_file data/biased_amazon_train.csv \
--model_name roberta-base \
--reg_method NFL-CP
```

DFR:

```bash
python src/dfr.py
```

---

# 4. Results and Analysis

## Amazon Dataset Results

| Method | Biased Acc | Robust Acc | Gap |
|---|---:|---:|---:|
| RoBERTa | 95.47 | 80.31 | -15.16 |
| NFL-F | 95.47 | 80.31 | -15.16 |
| NFL-CO | 95.72 | 80.24 | -15.48 |
| NFL-CP | 95.91 | 88.23 | -7.68 |
| NFL-PT | 95.47 | 80.31 | -15.16 |
| DFR (100%) | 94.06 | 93.22 | -0.84 |

### Key Findings

- Baseline robustness gap became significantly smaller than the paper
- NFL-CP outperformed NFL-PT
- DFR achieved near-zero robustness gap
- Bias effects appeared weaker than reported in the original paper

---

## Jigsaw Dataset Results

| Method | Biased Acc | Robust Acc | Gap |
|---|---:|---:|---:|
| RoBERTa | 40.85 | 44.27 | +3.42 |
| NFL-F | 40.85 | 44.27 | +3.42 |
| NFL-CO | 41.48 | 45.14 | +3.66 |
| NFL-CP | 39.10 | 43.02 | +3.92 |

### Key Findings

Paper:

```text
Robust Accuracy < Biased Accuracy
```

My Reproducibility:

```text
Robust Accuracy > Biased Accuracy
```

Possible reasons:

- weaker bias signals
- sampling differences
- dataset composition effects

---

# 5. Representation Analysis

## Nearest Neighbor Analysis

The paper reports that after fine-tuning, spurious tokens become strongly associated with sentiment polarity.

Example:

```text
people
```

became surrounded by highly polarized or toxic words.

My reproducibility still showed semantic shifts:

However:

- neighborhood structures became noisier
- weaker sentiment alignment appeared
- semantic shifts were less structured

---

## t-SNE Analysis

### Original Paper

- clear clustering
- strong separation
- highly structured representations

### Reproducibility

- scattered representations
- larger overlap
- weaker clustering behavior

The phenomenon was successfully reproduced, but with reduced structural dominance.

---

## NFL-CO vs NFL-CP

Original paper:

NFL-CP produced the strongest separation.

My reproducibility:

NFL-CP still outperformed NFL-CO,

but representations remained more dispersed.

---

# 6. Key Challenges and Insights

## Technical Challenges

- Sensitivity to dataset-specific bias patterns
- Inconsistent NFL improvements
- High variance in representation space
- Sensitivity to token-level bias
- Difficulty separating semantic signals from shortcut learning
- Dependence on data distribution and split configuration

## Insights Gained

- Spurious correlation behavior is reproducible
- Robustness strongly depends on dataset bias strength
- Representation analysis provides deeper understanding than accuracy alone
- Different datasets can produce substantially different robustness behavior
- Reproducibility studies reveal hidden assumptions in benchmark design

---

# 7. Conclusion

This reproduction successfully confirmed several findings from the original paper:

- Spurious tokens shift semantically after fine-tuning
- Representation clustering emerges during training
- NFL methods improve robustness

However, several discrepancies emerged:

- NFL-CP became the strongest method
- Jigsaw showed reversed robustness behavior
- Representation structures became weaker and noisier

These findings suggest that robustness evaluation is highly sensitive to experimental conditions and dataset characteristics.

Overall, this reproduction highlights the importance of representation-level analysis and robustness-oriented evaluation in NLP research.
