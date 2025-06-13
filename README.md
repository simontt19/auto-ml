# Auto-ML

An automated machine learning project repository that implements a complete ML pipeline from data ingestion to model deployment.

## Overview

This repository contains a robust, enterprise-grade machine learning framework that follows best practices for automated ML workflows. The pipeline includes:

- **Data Ingestion**: Automated loading and preprocessing of datasets
- **Feature Engineering**: Comprehensive data transformation and feature creation
- **Model Training**: Multiple algorithm training with automatic model selection
- **Evaluation**: Cross-validation and performance metrics
- **Deployment**: Model persistence and results logging

## Features

- 🚀 **Complete ML Pipeline**: End-to-end automation from data to deployment
- 📊 **Multiple Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, LightGBM
- 🔄 **Cross-Validation**: Robust model evaluation with stratified k-fold CV
- 📈 **Real Metrics**: AUC, accuracy, precision, recall, F1-score, log-loss
- 🛠️ **CLI Support**: Command-line interface with configurable options
- 📝 **Comprehensive Logging**: Detailed logs for debugging and monitoring
- 🧪 **Real Data**: Uses actual datasets (UCI Adult dataset) - no simulations

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/simontt19/auto-ml.git
   cd auto-ml
   ```

2. **Create and activate virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**
   ```bash
   mkdir -p data
   curl -o data/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
   curl -o data/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
   ```

### Usage

#### Basic Pipeline Execution

Run the complete ML pipeline with default settings:

```bash
python main_pipeline.py
```

#### Advanced Usage with CLI Options

```bash
# Specify custom data path
python main_pipeline.py --data-path /path/to/your/data/

# Set log level for debugging
python main_pipeline.py --log-level DEBUG

# Combine options
python main_pipeline.py --data-path data/ --log-level INFO
```

#### CLI Options

- `--data-path`: Path to the dataset directory (default: `data/`)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, ERROR (default: `INFO`)

## Pipeline Architecture

### 1. Data Ingestion (`data_ingestion.py`)

- Loads UCI Adult dataset
- Handles missing values and data cleaning
- Creates binary target variable
- Validates data integrity

### 2. Feature Engineering (`feature_engineering.py`)

- **Missing Value Imputation**: Mode for categorical, median for numerical
- **Categorical Encoding**: Label encoding with unseen category handling
- **Numerical Scaling**: StandardScaler for normalization
- **Feature Creation**: Age groups, education levels, work categories, wealth indicators

### 3. Model Training (`model_training.py`)

- **Multiple Algorithms**: 4 different models trained in parallel
- **Automatic Selection**: Best model chosen by AUC score
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Feature Importance**: Analysis of most predictive features

### 4. Pipeline Orchestration (`main_pipeline.py`)

- Coordinates all pipeline steps
- Handles error logging and recovery
- Generates comprehensive reports
- Saves results and models

## Output Files

### Logs

- `pipeline.log`: Detailed execution logs with timestamps
- `debug_logs.md`: Troubleshooting and debugging insights

### Results

- `training_results_YYYYMMDD_HHMMSS.json`: Model performance metrics
- Console output: Real-time progress and summary

### Example Output

```
============================================================
STARTING FULL ML PIPELINE
============================================================

==================== STEP 1: DATA INGESTION ====================
INFO:data_ingestion:Loading Adult dataset...
INFO:data_ingestion:Training data shape: (32561, 15)
INFO:data_ingestion:Test data shape: (16281, 15)

==================== STEP 2: FEATURE ENGINEERING ====================
INFO:feature_engineering:Starting feature engineering...
INFO:feature_engineering:Feature engineering completed. Final shape: (32561, 21)

==================== STEP 3: MODEL TRAINING & EVALUATION ====================
INFO:model_training:Starting model training and evaluation...
INFO:model_training:logistic_regression - AUC: 0.8553, Accuracy: 0.8285
INFO:model_training:random_forest - AUC: 0.9052, Accuracy: 0.8554
INFO:model_training:gradient_boosting - AUC: 0.9196, Accuracy: 0.8668
INFO:model_training:lightgbm - AUC: 0.9269, Accuracy: 0.8731
INFO:model_training:Best model selected: lightgbm (AUC: 0.9269)

==================== PIPELINE SUMMARY ====================
Data processed:
  - Training samples: 32561
  - Test samples: 16281
  - Total features: 19

Model Performance:
  lightgbm:
    - AUC: 0.9269
    - Accuracy: 0.8731
    - F1-Score: 0.7077

🎉 Pipeline completed successfully!
```

## Troubleshooting

### Common Issues

1. **Dataset not found**

   ```
   ERROR:data_ingestion:Error loading dataset: [Errno 2] No such file or directory: 'data/adult.data'
   ```

   **Solution**: Ensure the dataset files are downloaded to the `data/` directory.

2. **Memory issues with large datasets**
   **Solution**: Use smaller datasets or increase system memory.

3. **LightGBM installation issues**
   **Solution**: Install LightGBM separately: `pip install lightgbm`

### Debug Mode

For detailed debugging, run with DEBUG log level:

```bash
python main_pipeline.py --log-level DEBUG
```

This will show:

- Detailed data processing steps
- Model training progress
- Memory usage
- Performance metrics

### Log Analysis

Check `pipeline.log` for:

- Error messages with full tracebacks
- Performance metrics and timing
- Data validation results
- Model selection process

## Project Structure

```
auto-ml/
├── data/                          # Dataset directory
│   ├── adult.data                 # Training data
│   └── adult.test                 # Test data
├── venv/                          # Virtual environment
├── data_ingestion.py              # Data loading and preprocessing
├── feature_engineering.py         # Feature creation and transformation
├── model_training.py              # Model training and evaluation
├── main_pipeline.py               # Pipeline orchestration
├── requirements.txt               # Python dependencies
├── tasks.md                       # Task tracking and progress
├── debug_logs.md                  # Debugging and troubleshooting
├── pipeline.log                   # Execution logs
└── README.md                      # This file
```

## Contributing

1. Follow the task-based development approach
2. Each task should be documented in `tasks.md`
3. All changes must be tested with real data
4. Update `debug_logs.md` with any issues or insights
5. Maintain comprehensive logging and error handling

## License

MIT License

## Acknowledgments

- UCI Machine Learning Repository for the Adult dataset
- Scikit-learn, LightGBM, and other open-source ML libraries
