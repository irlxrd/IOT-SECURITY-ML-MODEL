# Network Traffic Classification using Machine Learning

This project implements machine learning models to classify network traffic into different attack types using the IoT-23 dataset.

## Project Overview

The system classifies network traffic into 5 categories:

- **Benign** - Normal network traffic
- **C&C** (Command & Control) - Botnet communication
- **DDoS** - Distributed Denial of Service attacks
- **PortScan** - Port scanning activities
- **Okiru** - Okiru malware traffic

## Dataset

The project uses the **IoT-23 dataset**, which contains labeled network traffic from IoT devices. The dataset is processed to create a balanced multiclass classification problem with approximately:

- 57,000 C&C samples (collected from all IoT-23 sub-datasets)
- Equal numbers of DDoS, Benign, PortScan, and Okiru samples
- **Total: ~285,000 balanced samples across 5 classes**

## Project Structure

```
project/
├── collect_c&c.py
├── train_models.py
├── balanced_multiclass.csv
├── web_app/
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── xgb_model.pkl
│   ├── mlp_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
└── plots/
    ├── feature_importance_comparison.png
    ├── model_accuracy_comparison.png
    ├── confusion_matrices.png
    ├── per_class_metrics.png
    ├── roc_curves.png
    ├── learning_curve_random_forest.png
    └── decision_tree.png
```

## Files Description

### collect_c&c.py

Data collection and preprocessing script that:

- Extracts C&C samples from all IoT-23 sub-datasets (~57k samples)
- Collects balanced samples of DDoS, Benign, PortScan, and Okiru from specific files
- Performs data cleaning and preprocessing
- Outputs a balanced dataset (`balanced_multiclass.csv`)

### train_models.py

Machine learning pipeline that:

- Trains 4 different classification models:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Multi-Layer Perceptron (MLP)
- Evaluates model performance with multiple metrics
- Generates comprehensive visualizations
- Saves trained models for deployment

## Installation

Install required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```

## Usage

### Step 1: Data Collection

```bash
python collect_c&c.py
```

This will create `balanced_multiclass.csv` with approximately 285k balanced samples.

### Step 2: Model Training

```bash
python train_models.py
```

This will:

- Train all 4 models
- Display classification reports and confusion matrices
- Generate 7 visualization plots in the `plots/` directory
- Save trained models in the `web_app/` directory

### Step 3: Modify Models (Optional)

Edit `train_models.py` to:

- Adjust hyperparameters
- Add new algorithms
- Modify feature engineering
- Change visualization settings

## Model Performance

All models achieve near-perfect classification accuracy (>99%) due to:

- Well-balanced dataset
- Distinct feature patterns between attack types
- High-quality labeled data from IoT-23

### Typical Results

| Model | Accuracy |
|-------|----------|
| Random Forest | ~100% |
| Gradient Boosting | ~100% |
| XGBoost | ~100% |
| MLP | ~98-99% |

## Visualizations

The training script generates 7 comprehensive visualizations:

1. **Feature Importance Comparison** - Top features across tree-based models
2. **Model Accuracy Comparison** - Bar chart comparing all models
3. **Confusion Matrices** - Heatmaps for each model showing prediction errors
4. **Per-Class Metrics** - Precision, Recall, and F1-Score by class
5. **ROC Curves** - One-vs-Rest ROC curves for multiclass classification
6. **Learning Curve** - Training vs validation performance
7. **Decision Tree** - Visual representation of a single tree from Random Forest

All plots are saved in the `plots/` directory.

## Features

The models use network flow features including:

- Protocol types (TCP, UDP, ICMP, etc.)
- Connection states
- Packet statistics (bytes, packets, duration)
- Flow characteristics
- Timing information

Categorical features (`proto`, `state`) are one-hot encoded, and missing values are handled appropriately.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- joblib

## Future Improvements

- Real-time traffic classification
- Deep learning models (LSTM, CNN)
- Feature selection optimization
- Model ensemble techniques
- Web interface for predictions
- Model interpretability (SHAP values)

## References

- **IoT-23 Dataset**: https://www.stratosphereips.org/datasets-iot23
- **Paper**: "A Labeled Dataset with Malicious and Benign IoT Network Traffic"

## License

[Add your license here]

## Contact

[Add your contact information]
