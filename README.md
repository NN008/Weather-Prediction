# Weather Prediction — ML Demo

A machine learning demo that predicts rainfall using atmospheric sensor data. The pipeline covers the full ML workflow — from exploratory data analysis through model training, active learning, and evaluation.

## Dataset
2,500 labeled observations across five meteorological features:
- Temperature, Humidity, Wind Speed, Cloud Cover, Pressure
- Binary target: `rain` / `no rain`
- No null values; near-symmetric distributions across all features

## Models
- **Random Forest Classifier** — ensemble of 20 decision trees with `warm_start` for incremental training
- **Logistic Regression** — linear baseline for comparison

Both models use an **active learning loop with uncertainty sampling**: at each epoch, the least confident prediction is added to the training set, improving performance with minimal labeled data.

## Results
| Model | Accuracy |
|---|---|
| Random Forest | ~99.5% |
| Logistic Regression | ~93.4% |

Random Forest nearly eliminated false negatives (3 misclassified rain samples). Results evaluated via accuracy, precision, recall, F1-score, and confusion matrices.

## Tech Stack
- Python, scikit-learn, pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## Structure
```
Weather-Prediction/
├── notebooks/
│   ├── Classification.ipynb    # Exploratory data analysis
│   └── Model_Compare.ipynb     # RF vs. Logistic Regression + active learning
└── data/
    ├── raw/
    │   └── Project1WeatherDataset.csv
    └── processed/
        └── weather_forecast_data.csv
```

## Usage
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook notebooks/Model_Compare.ipynb
```
