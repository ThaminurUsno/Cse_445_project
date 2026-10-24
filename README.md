# Dataset Inspection, Exploratory Data Analysis, and Baseline Model

## Project Title

**Obesity Level Detector: A Comparative Study of Anthropometric, Demographic, and Lifestyle Features in the Dhaka Obesity Dataset**

## 1. Overview

During Week 1, Person 1 focused on dataset preparation, exploratory data analysis (EDA), and the development of a Logistic Regression baseline model for obesity level classification.

The primary objective was to understand the dataset, identify data quality issues, explore feature distributions, prepare the data for machine learning, and establish an initial baseline for future model comparisons.

## 2. Week 1 Objectives

- Load and inspect the Dhaka Obesity Dataset.
- Examine dataset dimensions, column names, and data types.
- Identify missing values and duplicate records.
- Perform exploratory data analysis (EDA).
- Analyze the distribution of obesity level classes.
- Prepare input features and the target variable.
- Split the dataset into training and testing sets.
- Develop a preprocessing pipeline for numerical and categorical features.
- Train a Logistic Regression baseline classifier.
- Evaluate model performance using classification metrics.
- Perform stratified cross-validation on the training data.

## 3. Dataset Loading and Inspection

The dataset was loaded into a Pandas DataFrame for initial inspection and analysis.

### Tasks Performed

- Loaded the CSV dataset.
- Checked the number of rows and columns.
- Inspected column names and data types.
- Reviewed the first few records.
- Examined the overall dataset structure.
- Checked for missing values.
- Identified duplicate records.

### Initial Dataset Summary

| Property | Result |
|---|---|
| Dataset | Dhaka Obesity Dataset |
| Original Rows | 2,182 |
| Number of Columns | 17 |
| Target Variable | `Obesity level` |
| Missing Values | 0 |
| Duplicate Records Identified | 25 |
| Rows After Duplicate Removal | 2,157 |

> **Note:** These figures were reported during the initial experiment. Verify them against the final dataset and executed code before submission.

## 4. Data Quality Assessment

A data quality assessment was conducted to identify potential issues before model training.

### 4.1 Missing Value Analysis

The dataset was examined for missing values across all columns. The initial inspection reported no missing values.

### 4.2 Duplicate Record Analysis

Duplicate records were identified during the initial inspection. Duplicate rows were removed before the subsequent modeling workflow.

### 4.3 Data Preparation

The data preparation process included:

- Inspecting the dataset for quality issues.
- Removing duplicate records.
- Identifying numerical and categorical features.
- Separating the input features from the target variable.

## 5. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the characteristics and distributions of the dataset.

### 5.1 Target Variable Distribution

The distribution of the `Obesity level` target variable was examined to understand the number of observations belonging to each obesity classification category.

### 5.2 Numerical Feature Analysis

Numerical feature distributions were examined to understand their ranges, variability, and overall patterns.

### 5.3 Categorical Feature Analysis

Categorical variables were analyzed to understand the frequency of different categories and their representation within the dataset.

### 5.4 Correlation Analysis

A correlation heatmap was used to explore relationships between numerical features and identify potentially correlated variables.

### EDA Visualizations

The analysis included:

- Obesity level distribution chart.
- Numerical feature distribution plots.
- Categorical feature distribution plots.
- Numerical feature correlation heatmap.

These visualizations provided an initial understanding of the dataset and helped establish a foundation for further feature analysis.

## 6. Feature Preparation

The target variable was defined as:

`Obesity level`

All remaining eligible columns were treated as input features.

### Feature Preparation Steps

1. Separated the input features (`X`) from the target variable (`y`).
2. Identified numerical and categorical columns.
3. Prepared preprocessing steps for both feature types.
4. Ensured that preprocessing transformations were fitted using the training data.

## 7. Train-Test Split

The dataset was divided into training and testing sets to evaluate model performance on unseen observations.

### Configuration

| Parameter | Value |
|---|---|
| Training Set | 80% |
| Testing Set | 20% |
| Random State | 42 |
| Splitting Method | Stratified Train-Test Split |

Stratification was used to preserve the relative distribution of obesity level classes across the training and testing sets.

Based on the initially reported dataset size after duplicate removal, the split contained:

- **Training samples:** 1,725
- **Testing samples:** 432

> **Note:** Confirm these sample counts using the final executed notebook.

## 8. Data Preprocessing Pipeline

A preprocessing pipeline was designed to handle numerical and categorical features consistently.

### 8.1 Numerical Feature Preprocessing

The following transformations were applied:

- **Median Imputation:** Handles missing numerical values using the median of the training data.
- **Standard Scaling:** Uses `StandardScaler` to standardize numerical features.

### 8.2 Categorical Feature Preprocessing

The following transformations were applied:

- **Most-Frequent Imputation:** Handles missing categorical values using the most frequent category.
- **One-Hot Encoding:** Converts categorical variables into numerical representations.
- **Unknown Category Handling:** Uses `handle_unknown="ignore"` to handle categories not observed during training.

### 8.3 Benefits of the Pipeline

- Provides consistent preprocessing.
- Helps prevent data leakage by fitting transformations on training data.
- Supports numerical and categorical features.
- Makes the workflow reproducible.
- Simplifies the integration of additional machine learning algorithms.

## 9. Logistic Regression Baseline Model

Logistic Regression was selected as the initial baseline classifier for the multiclass obesity level classification task.

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Maximum Iterations | 3,000 |
| Random State | 42 |
| Classification Task | Multiclass Classification |
| Evaluation Methods | Holdout Testing and Cross-Validation |

The preprocessing pipeline and Logistic Regression classifier were combined into a single machine learning workflow.

The model was trained on the training set and evaluated on the held-out test set.

## 10. Model Evaluation

The baseline model was evaluated using several classification metrics.

### 10.1 Evaluation Metrics

- **Accuracy:** Measures the proportion of correctly classified observations.
- **Weighted Precision:** Calculates precision while accounting for the number of observations in each class.
- **Weighted Recall:** Calculates recall while accounting for class support.
- **Weighted F1-Score:** Measures the weighted balance between precision and recall.
- **Classification Report:** Provides precision, recall, F1-score, and support for individual classes.
- **Confusion Matrix:** Shows correct and incorrect predictions for each target class.

### 10.2 Initially Reported Test Results

| Metric | Reported Result |
|---|---:|
| Accuracy | 88.19% |
| Weighted Precision | 88.26% |
| Weighted Recall | 88.19% |
| Weighted F1-Score | 88.17% |

> **Important:** These results were reported from an earlier experiment. Verify them against the final executed notebook or saved evaluation outputs before presenting them as confirmed results.

## 11. Stratified Five-Fold Cross-Validation

Stratified five-fold cross-validation was used on the training data to evaluate baseline model performance across multiple data splits.

Stratification helps preserve the relative proportions of the target classes within each fold.

### Initially Reported Cross-Validation Results

| Metric | Mean Score | Standard Deviation |
|---|---:|---:|
| Accuracy | 89.86% | 3.36% |
| Weighted Precision | 89.92% | 3.39% |
| Weighted Recall | 89.86% | 3.36% |
| Weighted F1-Score | 89.83% | 3.36% |

The mean score summarizes performance across the five folds, while the standard deviation indicates the variation in scores between folds.

> **Important:** Verify these results against the final executed code and saved outputs before submission.

## 12. Tools and Technologies

The following tools and libraries were used or planned for the Week 1 workflow:

| Tool or Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Jupyter Notebook | Interactive development and experimentation |
| Git | Version control |
| GitHub | Project collaboration and code hosting |

## 13. Week 1 Deliverables

The Week 1 work covered the following deliverables:

- Dataset loading and inspection.
- Data quality assessment.
- Missing value and duplicate record analysis.
- Exploratory data analysis visualizations.
- Numerical and categorical feature identification.
- Train-test splitting strategy.
- Data preprocessing pipeline.
- Logistic Regression baseline model.
- Classification metrics and classification report.
- Confusion matrix.
- Stratified five-fold cross-validation.
- Saved notebook and relevant output files.

## 14. Conclusion

Week 1 established the initial data preparation and machine learning workflow for the obesity level classification project.

The work included dataset inspection, data quality assessment, exploratory data analysis, feature preparation, preprocessing, Logistic Regression baseline training, and model evaluation.

The baseline model provides an initial reference point for comparing additional machine learning algorithms in subsequent project stages.

Future work can extend this foundation by evaluating alternative classifiers, investigating feature importance, and applying explainability techniques to better understand the factors influencing model predictions.

## 15. Disclaimer

This project is intended for academic research and educational purposes. The model's predictions should not be interpreted as medical diagnoses or as a substitute for professional medical advice.
