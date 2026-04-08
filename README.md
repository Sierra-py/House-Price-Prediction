# House Price Prediction

Kaggle competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

**Leaderboard Score: 0.12331 | Rank: 661**

---

## Overview

End-to-end regression pipeline predicting residential home sale prices in Ames, Iowa. The dataset contains 79 features covering physical attributes, quality ratings, neighborhood, and sale conditions.

---

## Approach

### Data Preprocessing
- Dropped high-null columns: `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `MasVnrArea`
- Dropped near-zero variance column: `Utilities`
- Filled basement and garage categoricals with semantic null values (`NoBasement`, `NoGarage`)
- Imputed `LotFrontage` using neighborhood-median to preserve local context
- Imputed `Electrical` with mode
- Added `HasGarage` binary flag; filled `GarageYrBlt` nulls with 0

### Feature Engineering
- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- `HouseAge = YrSold - YearBuilt`
- `RemodAge = YrSold - YearRemodAdd`
- `HasGarage` — binary indicator for garage presence

### Encoding
- **Ordinal features** — `OrdinalEncoder` with explicit category orders (e.g. `Po < Fa < TA < Gd < Ex`)
- **Nominal features** — `OneHotEncoder` with `handle_unknown='ignore'`
- **Numeric features** — `StandardScaler` for linear models; passthrough for tree models

### Models
All models evaluated with 5-fold cross-validation on log-transformed `SalePrice`.

| Model | CV RMSE (log scale) |
|---|---|
| XGBoost | 0.1226 |
| LightGBM | 0.1299 |
| Lasso | 0.1359 |
| ElasticNet | 0.1406 |
| Ridge | 0.1410 |
| Random Forest | 0.1415 |

### Final Submission
Weighted blend of top three models:

```
Final = 0.6 * XGBoost + 0.2 * LightGBM + 0.2 * Lasso
CV RMSE: 0.1210
Kaggle Score: 0.12331
```

---

## Project Structure

```
House-Price-Prediction/
├── house_price_prediction.ipynb   # Main notebook
├── train.csv                      # Training data
├── test.csv                       # Test data
├── submission.csv                 # Final Kaggle submission
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Sierra-py/House-Price-Prediction
cd House-Price-Prediction
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in the root directory.

---

## Requirements

See `requirements.txt`.
