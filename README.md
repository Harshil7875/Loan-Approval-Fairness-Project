# Loan Approval Fairness Project

This repository addresses **fairness** in loan approvals by examining and mitigating biases in a machine-learning model trained on public HMDA (Home Mortgage Disclosure Act) data. We develop two approaches: a **baseline neural network** and an **adversarially debiased** model that reduces discrimination against protected groups (race, gender, ethnicity) and socio-economic backgrounds.

## Repository Structure

```
.
├── ECS289G_Term_Project
│   ├── data_cleaning.ipynb
│   ├── data_analysis.ipynb
│   ├── data_transform.ipynb
│   ├── baseline_model.ipynb
│   └── adversarial_model.ipynb
├── data.zip
├── docs
│   ├── ...
├── dump
│   ├── ...
└── test.py
```

1. **`ECS289G_Term_Project`**  
   - **`data_cleaning.ipynb`**  
     - **Goal**: Load raw HMDA data (in CSV, from `data.zip`) and **select key columns** (demographic, SES, loan attributes, `action_taken`).  
     - Handles missing values: imputing `interest_rate`, preserving demographic info with placeholders.  
     - One-hot encodes categorical features and **saves** a clean CSV.
   
   - **`data_analysis.ipynb`**  
     - **Goal**: Perform **exploratory data analysis (EDA)**.  
     - Examines the distribution of numeric features, correlation matrix, and class imbalance in `action_taken`.  
     - Offers **insights** into potential biases or data skews.
   
   - **`data_transform.ipynb`**  
     - **Goal**: **Feature engineering** and **normalization**.  
     - Classifies each row into Low/Middle/High SES via `tract_to_msa_income_percentage`.  
     - Uses `MinMaxScaler` on numeric columns for stable neural-network training.  
     - Produces a **transformed dataset** with an additional `SES_group` column.
   
   - **`baseline_model.ipynb`**  
     - **Goal**: Train a **basic neural network** for binary classification (loan approved vs. not).  
     - Maps `action_taken` → `loan_approved` (1 or 0), splits into train/test, and scales features if needed.  
     - Uses **Keras** with dense layers, dropout, batch normalization, and early stopping.  
     - Evaluates **accuracy, precision, recall, F1**, plus **fairness metrics** (Statistical Parity, Predictive Parity, Equal Opportunity) by race, gender, ethnicity, and SES group.  
     - Baseline results show **high accuracy** (~91.8%) but disparities in approvals across some subgroups.
   
   - **`adversarial_model.ipynb`**  
     - **Goal**: Implement **adversarial debiasing** using a **Gradient Reversal Layer**.  
     - Separates **sensitive attributes** (race, gender, ethnicity) and **resamples** the training data to address subgroup imbalances.  
     - Trains a main model for approval predictions + adversary networks predicting sensitive traits (race, gender, ethnicity). The GRL penalizes the main network if it encodes sensitive info.  
     - Achieves lower raw accuracy (~82.6%) but **improved fairness** across subgroups (less disparity).

2. **`data.zip`**  
   - Contains raw HMDA data used in `data_cleaning.ipynb`.

3. **`docs` Folder**  
   - Contains PDF files (e.g., `Step_2.pdf`, `important_questions.pdf`) with **supplementary details** about the methodology, steps, and questions driving this project.

4. **`dump` Folder**  
   - Stores text versions of the notebooks or logs from earlier runs.

5. **`test.py`**  
   - A placeholder Python script (possibly for quick testing or debugging).

---

## Data & Modeling Pipeline

**Data Source**: HMDA loan-level data, including race, ethnicity, sex, income, property, and loan status. The target is `action_taken`, mapped to binary approval.

1. **`data_cleaning.ipynb`** → cleans & selects key columns → outputs **cleaned dataset**  
2. **`data_analysis.ipynb`** → optional EDA  
3. **`data_transform.ipynb`** → adds `SES_group` and scales numeric features → **transformed dataset**  
4. **`baseline_model.ipynb`** → trains a **straightforward neural net**; sees high accuracy but fairness concerns  
5. **`adversarial_model.ipynb`** → addresses bias via **adversarial training** + group-specific resampling

---

## Key Results

- **Baseline Model**:  
  - ~91.8% accuracy, ~0.91 recall (TPR).  
  - Demonstrates **sizable** differences in approval rates or TPR across certain race/SES groups.
- **Adversarial Debiasing Model**:  
  - ~82.6% accuracy, ~0.77 recall.  
  - **Better fairness** across race/gender/ethnicity/SES subgroups, though at the cost of some predictive performance.  
  - Illustrates the **performance–fairness trade-off** commonly seen in debiasing techniques.

---

## Usage Instructions

1. **Environment**: Install Python 3.x plus libraries (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `imblearn`, etc.).  
2. **Run Order**:
   - **`data_cleaning.ipynb`**: Points to raw CSV from `data.zip`; outputs `cleaned_dataset.csv`.  
   - **`data_analysis.ipynb`**: Optional EDA.  
   - **`data_transform.ipynb`**: Scales numeric columns, sets `SES_group`; produces `transformed_dataset.csv`.  
   - **`baseline_model.ipynb`**: Trains the baseline neural network; prints performance/fairness.  
   - **`adversarial_model.ipynb`**: Trains the adversarially debiased model; prints new fairness metrics.

---

## Documentation & Context

- This entire codebase was developed as a **term project** for **ECS 289G 001: Artificial Intelligence**, a **graduate-level special topic course** at the **University of California, Davis in Fall 2024**.
- The `docs/` folder contains PDF files elaborating the steps (2, 3, 4, 5), relevant **research questions**, and a broader discussion on the **ethics** of AI in lending.

---

## Thank you for exploring the **Loan Approval Fairness Project**!