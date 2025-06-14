# Project Context: Adult Income Prediction

## Current State

**Status**: Initial setup phase
**Last Updated**: 2024-06-14
**Phase**: Data exploration and preprocessing

## Data Sources

### Primary Dataset

- **Name**: UCI Adult Income Dataset
- **Location**: `data/adult_income.csv`
- **Format**: CSV with 14 features + target variable
- **Size**: 48,842 records
- **Quality**: Generally good, some missing values

### Data Characteristics

- **Numerical Features**: 6 (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week)
- **Categorical Features**: 8 (workclass, education, marital-status, occupation, relationship, race, sex, native-country)
- **Target Variable**: income (binary: >50K, <=50K)

## Previous Experiments

None yet - this is the initial project setup.

## Known Issues

1. **Missing Values**: Some categorical features have missing values (marked as "?")
2. **Class Imbalance**: Target variable is imbalanced (~24% >50K, ~76% <=50K)
3. **Feature Correlation**: Some features may be highly correlated
4. **Data Quality**: Need to validate data consistency and outliers

## Constraints

### Technical Constraints

- Model must be interpretable (SHAP values, feature importance)
- Prediction time < 2 seconds
- Memory usage < 1GB during training
- Must handle missing values gracefully

### Business Constraints

- Model should provide business insights
- Results must be explainable to non-technical stakeholders
- Deployment must be production-ready
- Model should be retrainable with new data

## Environment

- **Framework**: Auto ML Framework
- **Python Version**: 3.8+
- **Key Libraries**: scikit-learn, pandas, numpy, matplotlib
- **Deployment**: FastAPI-based REST API

## Next Steps

1. **Data Exploration**: Analyze data quality, distributions, and relationships
2. **Data Preprocessing**: Handle missing values and feature engineering
3. **Baseline Model**: Train simple baseline model for comparison
4. **Feature Engineering**: Create meaningful features for better performance
