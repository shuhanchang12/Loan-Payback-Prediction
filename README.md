# 🏦 Loan Payback Prediction - Binary Classification ML Project

## 📋 Project Overview

**Objective:** Build a machine learning model to predict whether a customer will pay back their loan.

**Competition:** [Kaggle Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)

**Problem Type:** Binary Classification

**Dataset:**
- Training set: 593,994 samples with 13 features
- Test set: 254,569 samples with 12 features
- Target variable: `loan_paid_back` (0 = Default, 1 = Paid Back)

---

## 🎯 Key Results

### Model Performance Comparison

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 1st | **Logistic Regression** | **89.98%** | 90.66% | 97.50% | 93.95% | **0.9103** |
| 🥈 2nd | **Random Forest (deployed)** | 89.95% | 90.30% | 97.93% | 93.96% | 0.8990 |
| 🥉 3rd | Polynomial Logistic | 89.58% | 90.64% | 96.96% | 93.69% | 0.8836 |
| 4th | Decision Tree | 89.45% | 90.43% | 97.06% | 93.63% | 0.8693 |
| 5th | KNN | 87.73% | 88.02% | 97.97% | 92.73% | 0.8093 |
| ⚠️ Baseline | Dummy Classifier | 79.88% | 79.88% | 100.00% | 88.81% | 0.5000 |

### Final Model Performance

**Selected Model:** Random Forest with 200 estimators

| Metric | Value |
|--------|-------|
| Accuracy | 90.25% |
| Precision | 90.77% |
| Recall | 97.75% |
| F1-Score | 94.13% |
| ROC-AUC | 0.9084 |

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
- **Log transformation** of `annual_income` (reduce skewness from 11+ to 0.7)
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
- Fixed random state (42)
- Stratified sampling for class balance preservation
- Complete preprocessing documentation

✅ **Comprehensive Evaluation**
- 5 different metrics per model
- Confusion matrices and ROC curves
- Business impact analysis

✅ **Interactive Visualizations**
- 5 Plotly dashboards
- Red-blue color scheme for accessibility
- Hover tooltips and zoom/pan functionality

✅ **Interpretability Focus**
- Misclassification analysis
- Decision tree decision paths
- Feature importance tracking

✅ **Production Ready**
- Error handling for missing values
- Scalable feature engineering pipeline
- Comprehensive evaluation function

---

## 💡 Key Insights

### 1. **Feature Engineering Matters**
- Log transformation of annual_income critical (skewness: 11+ → 0.7)
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

### For Immediate Deployment
1. Deploy Random Forest model with threshold at 0.5
2. Monitor prediction probabilities
3. Set decision threshold based on business risk tolerance
4. Track model performance on new data

### For Future Improvements
1. Implement class weight balancing based on business costs
2. Collect feature importance feedback from credit officers
3. Perform fairness audits for demographic bias
4. Set up continuous monitoring dashboard

### For Production Robustness
1. Implement automated retraining pipeline (monthly/quarterly)
2. Add fairness checks and demographic monitoring
3. Create model explainability dashboard for credit decisions
4. Document all preprocessing steps for compliance

---

## 📈 Performance Trends

**Accuracy Distribution:** 79.88% - 89.98%
- Baseline (Dummy): 79.88% (pure majority class)
- All trained models: 87.73% - 89.98% (significant improvement!)

**Recall Comparison:** 96.96% - 100%
- High recall across all models (catch most defaults)
- Trade-off: Higher false positive rate (more caution needed)

**ROC-AUC Ranking:** 0.5 - 0.9103
- Logistic Regression best discriminator (0.9103)
- Random Forest strong performer (0.8990)
- Baseline provides sanity check (0.5000)

---

## 📝 Data Dictionary

### Features (13 original, 55 engineered)

| Feature | Type | Description |
|---------|------|-------------|
| id | int | Unique identifier |
| annual_income | float | Yearly income (log-transformed in pipeline) |
| debt_to_income_ratio | float | Debt-to-income ratio |
| credit_score | int | Credit score (300-850) |
| loan_amount | float | Requested loan amount |
| interest_rate | float | Applied interest rate |
| gender | categorical | Female / Male |
| marital_status | categorical | Single / Married / Divorced |
| education_level | categorical | High School / Bachelor's / Master's |
| employment_status | categorical | Employed / Self-employed / Unemployed |
| loan_purpose | categorical | Personal / Debt consolidation / Home / Other |
| grade_subgrade | categorical | A1-F5 (loan grade) |
| **loan_paid_back** | **target** | **0 = Default, 1 = Paid Back** |

---

## 🔗 Related Resources

- **Kaggle Competition:** https://www.kaggle.com/competitions/playground-series-s5e11
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Plotly Interactive Visualization:** https://plotly.com/python/
- **Data Science Best Practices:** https://towardsdatascience.com/

---

## 📞 Contact & Support

**Project Team:** Shuhan Chang, Guillaume Letosser, Tristant deMaricourt

**Questions or Issues?**
- Check notebook comments and docstrings
- Review error messages in execution outputs
- Verify data files (train.csv, test.csv) exist in correct directory

---

## 📜 License

This project is part of the Kaggle Playground Series. 
Data and competition rules available at: https://www.kaggle.com/competitions/playground-series-s5e11

---

## ✨ Highlights

🎓 **Educational Value:**
- Complete ML pipeline from EDA to deployment
- Model comparison and evaluation best practices
- Production-ready code patterns
- Interpretability and explainability focus

📊 **Practical Implementation:**
- 6 different algorithms compared
- Interactive visualizations with Plotly
- Comprehensive error analysis
- Business impact quantification

🚀 **Deployment Ready:**
- Reproducible preprocessing pipeline
- Scalable feature engineering
- Complete evaluation framework
- Production considerations documented

---

**Last Updated:** December 7, 2025
**Status:** ✅ Complete and Ready for Production
**Notebook Version:** Optimized with English transitions and interactive dashboard
