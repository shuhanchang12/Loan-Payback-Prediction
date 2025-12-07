# 🏦 Loan Payback Prediction - Binary Classification ML Project

## 🎯 Executive Summary

**Problem:** Predict whether a loan customer will pay back their loan in a Kaggle competition.

**Solution:** Built and compared 6 machine learning models on 593,994 training samples.

**Results:**
- ✅ Best Accuracy: **89.98%** (Logistic Regression)
- ✅ Best ROC-AUC: **0.9103** (Logistic Regression) ⭐ Excellent generalization
- ✅ Deployed Model: **Random Forest** (89.95% accuracy, 97.93% recall)
- ✅ All models outperform baseline (79.88%) by **8-10 percentage points**

---

## 📋 Project Overview

**Objective:** Build a machine learning model to predict whether a customer will pay back their loan.

**Competition:** [Kaggle Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)

**Problem Type:** Binary Classification

**Dataset:**

- Training set: 593,994 samples with 13 features
- Test set: 254,569 samples with 12 features
- Target variable: `loan_paid_back` (0 = Default, 1 = Paid Back)
- **Class Balance:** 79.88% paid back, 20.12% default (moderate imbalance)

---

## 🎯 Key Results

### Model Performance Comparison

| Rank          | Model                              | Accuracy         | Precision | Recall  | F1-Score | ROC-AUC          |
| ------------- | ---------------------------------- | ---------------- | --------- | ------- | -------- | ---------------- |
| 🥇 1st        | **Logistic Regression**      | **89.98%** | 90.66%    | 97.50%  | 93.95%   | **0.9103** |
| 🥈 2nd        | **Random Forest (deployed)** | 89.95%           | 90.30%    | 97.93%  | 93.96%   | 0.8990           |
| 🥉 3rd        | Polynomial Logistic                | 89.58%           | 90.64%    | 96.96%  | 93.69%   | 0.8836           |
| 4th           | Decision Tree                      | 89.45%           | 90.43%    | 97.06%  | 93.63%   | 0.8693           |
| 5th           | KNN                                | 87.73%           | 88.02%    | 97.97%  | 92.73%   | 0.8093           |
| ⚠️ Baseline | Dummy Classifier                   | 79.88%           | 79.88%    | 100.00% | 88.81%   | 0.5000           |

### Final Model Performance

**Selected Model:** Random Forest with 200 estimators

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 90.25% |
| Precision | 90.77% |
| Recall    | 97.75% |
| F1-Score  | 94.13% |
| ROC-AUC   | 0.9084 |

### Business Impact

- ✅ **True Positives:** 3,126 good loans correctly approved
- ⚠️ **False Positives:** 318 risky loans incorrectly approved
- ❌ **False Negatives:** 72 good loans incorrectly rejected
- ✓ **True Negatives:** 733 risky loans correctly rejected
- **Overall Error Rate:** 9.75%

---

## 📁 Project Structure

```
DL_Predicting Loan Payback/
├── Loan_Payback_Prediction.ipynb          # Main notebook (fully optimized)
├── loan.py                                  # Python script (alternative)
├── train.csv                                # Training data (593,994 samples)
├── test.csv                                 # Test data (254,569 samples)
├── submission_final.csv                     # Final predictions (254,569 rows)
├── README.md                                # This file
└── github/
    └── Loan_Payback_Prediction.ipynb        # Optimized notebook version
```

---

## 🔍 Notebook Sections

### 1️⃣ **Introduction & Project Goal**

- Problem statement and business context
- Focus on interpretability and robustness
- Principles: reproducibility, realistic evaluation, business context

### 2️⃣ **Setup & Libraries**

- Import required packages (scikit-learn, pandas, plotly, seaborn)
- Configure environment settings
- Set random state for reproducibility (RANDOM_STATE = 42)

### 3️⃣ **Data Loading**

- Load training and test datasets
- Display dataset shapes and sample data
- Verify data integrity

### 4️⃣ **Exploratory Data Analysis (EDA)**

- Target distribution analysis (class imbalance: 80% vs 20%)
- Feature distributions and skewness
- Correlation analysis with target
- Missing value patterns

### 5️⃣ **Feature Engineering & Preprocessing**

- **Log transformation** of `annual_income` (reduce skewness from 1.72 to -0.34)
- **One-hot encoding** of categorical variables (6 categories)
- **Imputation** of missing values (median for numeric, mode for categorical)
- **Scaling** via StandardScaler in model pipelines
- **Output:** 55 engineered features from 13 original

### 6️⃣ **Model Training & Comparison**

- Train 6 models: Dummy, Logistic, Polynomial Logistic, Decision Tree, Random Forest, KNN
- Stratified sampling (20K from 593,994) to preserve class balance
- 80/20 train-validation split
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### 7️⃣ **Interactive Model Comparison Dashboard**

- **5 interactive Plotly visualizations:**
  1. Bar Chart (all metrics across models)
  2. Radar Chart (multi-dimensional profile)
  3. Heatmap (performance matrix, red-blue colorscale)
  4. Scatter Plot (Accuracy vs ROC-AUC)
  5. Ranking Table (weighted composite score)
- Composite score: 30% Accuracy + 20% Precision + 20% Recall + 20% F1 + 10% ROC-AUC

### 8️⃣ **Detailed Model Evaluation**

- Classification reports (per-class precision, recall, F1)
- Confusion matrices with visualizations
- ROC curves and threshold analysis
- Business impact analysis (TN, FP, FN, TP)

### 9️⃣ **Interpretability & Error Analysis**

- Misclassified sample examination
- Decision tree decision paths
- Feature importance analysis
- Understanding model failure modes

### 🔟 **Final Model & Submission**

- Retrain Random Forest on full training dataset (593,994 samples)
- Generate predictions on test set (254,569 samples)
- Export to `submission_final.csv`
- Production considerations and best practices

### 1️⃣1️⃣ **Comprehensive Model Evaluation Function**

- Utility function for complete performance assessment
- Outputs: metrics, classification report, confusion matrix, business impact
- Useful for ongoing monitoring and decision-making

### 1️⃣2️⃣ **Key Findings & Recommendations**

- Model rankings and comparative analysis
- Key insights from data exploration
- Actionable recommendations for deployment
- Future improvement suggestions

---

## 🛠️ Technologies & Libraries

**Core ML:**

- scikit-learn (6 models, preprocessing, metrics)
- pandas (data manipulation)
- numpy (numerical computing)

**Visualization:**

- Plotly (interactive dashboards)
- Matplotlib & Seaborn (static plots)

**Data Processing:**

- StandardScaler (feature scaling)
- PolynomialFeatures (non-linear transformations)
- Pipeline (reproducible workflows)

**Python Version:** 3.12.12

---

## 🚀 How to Use

### Option 1: Run the Complete Notebook

```bash
# Navigate to the notebook directory
cd DL_Predicting\ Loan\ PayBack/github/

# Open with Jupyter
jupyter notebook Loan_Payback_Prediction.ipynb
```

**Expected execution time:** ~31 seconds for full run

### Option 2: Run Specific Sections

Each section is independent. Execute cells in order:

1. Setup & Libraries
2. Data Loading
3. EDA (optional for exploratory insights)
4. Feature Engineering
5. Model Training
6. Dashboard (optional for visualization)
7. Final Model Training & Submission

### Option 3: Use as Python Module

```python
# Import the prepare_features function
from Loan_Payback_Prediction import prepare_features

# Prepare your data
X, y, X_test, test_ids = prepare_features(df_train, df_test)

# Train your own model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
```

---

## 📊 Key Features

✅ **Reproducible Pipeline**

- Fixed random state (RANDOM_STATE = 42) for deterministic results
- Stratified sampling preserves class balance (79.88% / 20.12%)
- Complete preprocessing documentation and feature engineering pipeline
- All 593,994 training samples processed consistently

✅ **Comprehensive Evaluation**

- 5 different metrics per model (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrices with visualizations
- ROC curves for threshold optimization
- Business impact analysis (TN/FP/FN/TP breakdown)

✅ **Interactive Visualizations**

- 5 Plotly dashboards (Bar Chart, Radar, Heatmap, Scatter, Table)
- Red-blue color scheme for accessibility
- Full zoom, pan, and export functionality
- Hover tooltips for detailed data inspection

✅ **Interpretability Focus**

- Misclassification analysis with sample examination
- Decision tree decision paths showing reasoning chain
- Feature importance tracking and visualization
- Per-model decision boundary analysis

✅ **Production Ready**

- Error handling for missing values
- Scalable feature engineering pipeline
- Comprehensive evaluation function for monitoring
- Modular design for easy adaptation

---

## 💡 Key Insights

### 1. **Feature Engineering Matters**

- Log transformation of annual_income critical (skewness: 1.72 → -0.34)
- One-hot encoding sufficient for categorical variables
- Feature scaling essential for distance-based and linear models

### 2. **Model Selection Trade-offs**

- **Logistic Regression:** Best generalization (highest ROC-AUC: 0.9103)
- **Random Forest:** Best recall (97.93%), chosen for production
- **Decision Tree:** Most interpretable (89.45% accuracy)
- **KNN:** Underperforms on high-dimensional data

### 3. **Class Imbalance Handling**

- Dataset: 80% positive (paid back) vs 20% negative (default)
- Stratified sampling preserves proportions
- Consider cost-based weighting for future improvements

### 4. **Error Patterns**

- Misclassifications often occur at feature boundaries
- Model confident on clear cases, uncertain on borderline cases
- Specific customer profiles more error-prone

---

## 🎯 Recommendations

### ✅ For Immediate Deployment

1. **Primary Model: Random Forest**
   - Deploy Random Forest (200 estimators) for production
   - Achieves 97.93% recall → catches most defaults
   - Set decision threshold at 0.5 (can be tuned based on cost)
   - Monitor prediction probabilities in real-time

2. **Backup Model: Logistic Regression**
   - Keep Logistic Regression as backup/fallback
   - Superior ROC-AUC (0.9103) indicates better generalization
   - Faster inference and more interpretable coefficients
   - Use for fairness audits and explanation generation

3. **Real-Time Monitoring**
   - Track daily accuracy and recall on new data
   - Alert if ROC-AUC drops below 0.88
   - Monitor feature distributions for data drift

### 🔧 For Performance Optimization

1. **Probability Calibration**
   - Apply Platt scaling to calibrate prediction probabilities
   - Improves decision-making at different thresholds

2. **Threshold Optimization**
   - Default threshold = 0.5 (balanced approach)
   - Increase to 0.6-0.7 for conservative approval
   - Decrease to 0.3-0.4 for aggressive approval
   - Choose based on FP/FN cost ratio

3. **Hyperparameter Tuning**
   - Random Forest: n_estimators 200-500, max_depth 10-20
   - Logistic: test different regularization (C parameter)
   - Use GridSearchCV with cross-validation

### 🚀 For Production Robustness

1. **Automated Retraining Pipeline**
   - Retrain monthly with new data
   - Compare new model performance vs baseline
   - A/B test before full deployment

2. **Fairness & Compliance**
   - Audit for demographic bias (gender, age, etc.)
   - Document decision rules for regulatory compliance
   - Create explainability reports for rejected loans
   - Track disparate impact metrics

3. **Monitoring Dashboard**
   - Track accuracy, precision, recall daily
   - Monitor false positive/negative rates
   - Alert on data drift or model degradation
   - Maintain audit log of all decisions

### 📈 For Future Improvements

1. **Feature Engineering**
   - Create interaction features (income × debt_ratio)
   - Engineer temporal features if available
   - Domain-speci