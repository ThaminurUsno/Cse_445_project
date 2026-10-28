Dhaka Obesity Analysis

An end-to-end data analysis and machine learning pipeline in a single Jupyter notebook. It explores a survey dataset of people in Dhaka and builds models to predict obesity level from body measurements, lifestyle and demographics.

Highlights
Full workflow in one notebook: load, inspect, clean, engineer features, explore, preprocess, model, evaluate, interpret, save.
Compares Logistic Regression, Random Forest and Gradient Boosting using 5-fold cross-validation and a stratified 80/20 train/test split.
Trains two setups to separate a trivial result from an informative one (see Key findings).
Saves charts, the cleaned dataset, a results table and trained models.
Dataset

Dhaka_Obesity.csv has 2,182 rows and 17 columns with no missing values. After removing 25 exact duplicates, 2,157 rows remain.

Group	Columns
Demographics and body	Gender, Age, Height (m), Weight (kg)
Eating habits	Family history of overweight, High caloric food consumption, Vegetable consumption frequency, Daily main meals frequency, Between-meal food consumption frequency, Monitor calories
Lifestyle	Smoking, Alcohol intake, Daily water intake, Physical exercise, Daily device usage duration, Mode of transportation
Target	Obesity level

Target classes: Insufficient_Weight, Normal_Weight, Overweight, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III.

The dataset file is not included in this repository. Add your own copy of Dhaka_Obesity.csv and make sure you have the right to share it before committing it.

Notebook workflow
Step	What it does
Setup	Installs and checks dependencies (run once)
Load and inspect	Shape, dtypes, missing values, duplicates, category listing
Clean	Strips whitespace, drops duplicates, checks for implausible age, height and weight values, flags age outliers
Feature engineering	Computes BMI and verifies labels against the standard WHO BMI cut-offs
EDA	Class balance, numeric distributions, correlations, obesity level by gender, age group and each habit, Cramér's V association ranking
Preprocessing	Ordinal encoding for ordered categories, one-hot encoding for unordered ones, standardized numeric features
Modeling	Three classifiers, each in a scikit-learn Pipeline, cross-validated and tested
Evaluation	Classification reports and confusion matrices for the best models
Interpretation	Permutation feature importance for the lifestyle-only model
Export	Cleaned CSV, results table, charts and models saved to outputs/
Inference	Example of predicting obesity level for a new person
Key findings
The labels are derived from BMI. About 99.9% of rows match the standard BMI cut-offs, so any model that sees height and weight is near-perfect by construction.
Model A (with height, weight and BMI): about 99.8% test accuracy. This mainly confirms the label is BMI-driven.
Model B (lifestyle and demographics only): about 66% test accuracy with Gradient Boosting. This is the more meaningful result, since it shows how much habits and demographics alone can explain.
Family history of overweight shows a strong association with obesity level in the exploratory analysis.

Exact figures can vary slightly with library versions. Run the notebook to reproduce them.

Getting started
Requirements
Python 3.9 or newer (3.11 or 3.12 recommended)
Packages: jupyter, numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, joblib
Installation
bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install jupyter numpy pandas matplotlib seaborn scikit-learn scipy joblib
python -m ipykernel install --user --name obesity-env --display-name "Python (obesity-env)"
Run
Place Dhaka_Obesity.csv next to the notebook, or edit the candidates list in the data-loading cell to point to your file.
Start Jupyter:
bash
   jupyter notebook
Open Dhaka_Obesity_Analysis.ipynb, select the Python (obesity-env) kernel, and choose Run → Run All Cells.

The setup cell at the top only needs to run once. Skip it afterwards.

Outputs

Everything is written to an outputs/ folder next to the notebook:

File	Description
Dhaka_Obesity_cleaned.csv	Cleaned dataset with BMI and age group
model_results.csv	Cross-validation and test metrics for all models
model_lifestyle_only.joblib	Best lifestyle-only pipeline
model_full.joblib	Best pipeline using height, weight and BMI
*.png	Charts: class distribution, distributions, habits vs obesity, Cramér's V, model comparison, confusion matrices, feature importance
Using a saved model
python
import joblib
import pandas as pd

model = joblib.load("outputs/model_lifestyle_only.joblib")

person = pd.DataFrame([{
    "Age": 27,
    "Vegetable consumption frequency": "Sometimes",
    "Daily main meals frequency": "Three",
    "Between-meal food consumption frequency": "Frequently",
    "Alcohol intake": "I do not drink",
    "Daily water intake": "Between 1 and 2 L",
    "Physical exercise": "I do not have",
    "Daily device usage duration": "More than 5 hours",
    "Gender": "Male",
    "Family history of overweight": "Yes",
    "High caloric food consumption": "Yes",
    "Smoking": "No",
    "Monitor calories": "No",
    "Mode of transportation": "Private Car",
}])

print(model.predict(person)[0])
Suggested repository layout
.
├── Dhaka_Obesity_Analysis.ipynb
├── README.md
├── .gitignore
├── data/                # place Dhaka_Obesity.csv here (not committed)
└── outputs/             # generated by the notebook

Suggested .gitignore:

venv/
.ipynb_checkpoints/
__pycache__/
outputs/*.joblib
data/*.csv
.DS_Store
Limitations
The target is a deterministic function of BMI, so high accuracy with height and weight is not evidence of predictive power.
The sample skews young (median age around 24), so findings may not generalize to older adults.
The data is observational and self-reported habits are associations, not causes.
This project is for analysis and learning. It is not a medical tool and should not be used for diagnosis.
License

Add a license of your choice (for example MIT) and state the dataset's source and terms here.
