# 🏦 Loan Payback Prediction: From EDA to Deployment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Team Members:** Shuhan Chang, Guillaume Letosser, Tristan de Maricourt

A comprehensive machine learning project for predicting loan payback using the [Kaggle Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11) dataset. This project emphasizes not just model performance, but **understanding model behavior** and **interpretable AI**.

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results &amp; Insights](#-results--insights)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

This project tackles a **binary classification problem**: predicting whether a customer will pay back their loan (`loan_paid_back`). Beyond achieving high accuracy, we focus on:

- **Model Interpretability**: Understanding why models make certain predictions
- **Baseline Comparison**: Why a Dummy Classifier matters
- **Feature Engineering**: Log transformations, scaling, and encoding strategies
- **Model Comparison**: From simple baselines to ensemble methods
- **Per-Sample Explanations**: Decision paths and probability reasoning

## ✨ Key Features

- 📊 **Comprehensive EDA**: Target balance analysis, feature distributions, correlation studies
- 🔧 **Feature Engineering**: Log transformations for skewed features, one-hot encoding
- 🤖 **Multiple Models**: Dummy, Logistic Regression, Polynomial Logistic, Decision Tree, Random Forest, KNN
- 📈 **Detailed Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC metrics
- 🔍 **Model Interpretability**: Confusion matrices, ROC curves, feature importance
- 💡 **Per-Sample Analysis**: Misclassification examples with explanations
- 🌳 **Interactive Visualizations**: Plotly-based decision tree and feature importance charts

## 📊 Dataset

**Source**: [Kaggle Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)

**Training Set**: ~165,000 examples
**Test Set**: Available for predictions

**Key Features**:

- **Numeric**: `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`
- **Categorical**: `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`
- **Target**: `loan_paid_back` (0 = Not Paid, 1 = Paid)

**Class Distribution**: Imbalanced (80% Paid, 20% Not Paid)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/shuhanchang12/Loan-Payback-Prediction.git
cd Loan-Payback-Prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
plotly>=5.0.0
jupyter>=1.0.0
```

## 🚀 Usage

### Running the Notebook

```bash
jupyter notebook Loan_Payback_Prediction.ipynb
```

### Quick Start

```python
# Load data
import pandas as pd
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Run preprocessing
X, y, X_test, test_ids = prepare_features(df_train, df_test)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Generate predictions
predictions = model.predict(X_test)
```

## 📈 Model Performance

### Validation Set Results (N=20,000)

| Model                         | Accuracy         | Precision | Recall | F1   | ROC-AUC         |
| ----------------------------- | ---------------- | --------- | ------ | ---- | --------------- |
| **Logistic Regression** | **90.48%** | 0.92      | 0.96   | 0.94 | **0.915** |
| **Random Forest**       | **90.28%** | 0.91      | 0.97   | 0.94 | **0.907** |
| Polynomial Logistic           | 89.33%           | 0.91      | 0.95   | 0.93 | 0.885           |
| Decision Tree                 | 89.13%           | 0.91      | 0.95   | 0.93 | 0.859           |
| KNN                           | 88.05%           | 0.90      | 0.95   | 0.92 | 0.829           |
| Dummy (Baseline)              | 79.88%           | 0.00      | 0.00   | 0.00 | 0.500           |

### Key Insights

- ✅ All models significantly outperform the baseline (79.88% accuracy)
- ✅ Logistic Regression achieves best accuracy (90.48%) and ROC-AUC (0.915)
- ✅ Random Forest provides strong performance with better interpretability via feature importance
- ⚠️ KNN underperforms due to curse of dimensionality from one-hot encoding

## 📁 Project Structure

```
Loan-Payback-Prediction/
├── Loan_Payback_Prediction.ipynb  # Main notebook
├── train.csv                       # Training data
├── test.csv                        # Test data
├── submission_final.csv            # Final predictions
├── README.md                       # This file
├── requirements.txt                # Dependencies
└── images/                         # Visualizations (optional)
```

## 🔬 Methodology

### 1. Data Exploration

- Target balance analysis (80/20 split)
- Feature distributions (histograms, boxplots)
- Correlation analysis with target variable
- Skewness detection (annual_income: skew = 11+)

### 2. Feature Engineering

- **Log transformation**: Applied to `annual_income` (reduces skew from 11+ to 0.7)
- **Scaling**: StandardScaler for distance-based and linear models
- **Encoding**: One-hot encoding for categorical variables
- **Missing values**: Median imputation for numeric, sentinel values for categorical

### 3. Model Training

- **Baseline**: Dummy Classifier (most_frequent strategy)
- **Linear**: Logistic Regression, Polynomial Logistic (degree=2)
- **Tree-based**: Decision Tree (max_depth=10), Random Forest (n_estimators=100)
- **Distance-based**: KNN (k=5)

### 4. Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves
- **Interpretability**: Feature importance, decision paths, misclassification analysis

## 🔍 Results & Insights

### Feature Importance (Top 5)

1. `employment_status` (Unemployed vs Employed)
2. `credit_score`
3. `debt_to_income_ratio`
4. `interest_rate`
5. `annual_income_log`

### Model Comparison

| Model                         | Best For                                                                |
| ----------------------------- | ----------------------------------------------------------------------- |
| **Logistic Regression** | Interpretable coefficients, fast deployment, regulatory compliance      |
| **Random Forest**       | Best out-of-the-box performance, feature importance, robust to outliers |
| **Decision Tree**       | Transparent rules, per-sample explanations, auditing                    |
| **KNN**                 | Low-dimensional data only (not recommended for this dataset)            |

### Business Implications

- **False Positives** (approving risky borrowers): Higher financial risk
- **False Negatives** (rejecting good borrowers): Lost revenue, customer dissatisfaction
- **Recommendation**: Use probability thresholds to balance business objectives

## 🚀 Future Improvements

### Model Enhancement

- [ ] Hyperparameter tuning (Grid/Random Search with CV)
- [ ] Probability calibration (Platt scaling, Isotonic regression)
- [ ] Advanced ensembles (XGBoost, LightGBM, Stacking)
- [ ] SHAP values for better feature interpretability

### Data & Features

- [ ] External data sources (credit bureau data, macroeconomic indicators)
- [ ] Time-based features (seasonality, trends)
- [ ] Interaction features (credit_score × debt_to_income_ratio)

### Production Readiness

- [ ] A/B testing framework
- [ ] Model monitoring and drift detection
- [ ] API deployment (Flask/FastAPI)
- [ ] Fairness and bias analysis
- [ ] Continuous retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Shuhan Chang** 
- **Guillaume Letosser**
- **Tristan de Maricourt**

## 🙏 Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset
- [Scikit-learn](https://scikit-learn.org/) for excellent ML tools

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the team members directly.

---

**Last Updated**: November 2025
