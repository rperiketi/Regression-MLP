# Boston House Price Regression with MLP

Predicting median owner-occupied home values (MEDV) in Boston suburbs using a
Multilayer Perceptron (MLP), evaluated with 10-fold cross-validation in
TensorFlow/Keras.

## Overview

This project implements and compares several fully connected feedforward
neural network architectures on the classic Boston Housing dataset:

1. **Two hidden layer MLP** (64 → 32 units) — baseline model
2. **Single hidden layer MLP** — sweep over hidden unit counts (8, 16, 32, 64, 128)
3. **Arbitrary depth MLP** — a generalized builder tested on several depths

All models are trained with the Adam optimizer and MSE loss, and evaluated
using 10-fold cross-validation with per-fold standardization to avoid data
leakage.

## Dataset

- **Source file:** `housing.csv` (whitespace-separated, no header)
- **Samples:** 506
- **Features:** 13 (e.g. `CRIM`, `ZN`, `INDUS`, `NOX`, `RM`, `TAX`, `PTRATIO`, `LSTAT`, etc.)
- **Target:** `MEDV` — median value of owner-occupied homes, in $1000s

> **Note:** `housing.csv` is not included in this repo/output — place it in the
> working directory before running the script.

## Requirements

```bash
pip install tensorflow scikeras scikit-learn pandas numpy
```

Tested with TensorFlow 2.x (the script prints the installed version on run).

## Project Structure

```
.
├── dlassignment1.py   # Main python script
├── housing.csv         # Dataset 
└── README.md
```

## How to Run

```bash
python dlassignment1.py
```

The script runs sequentially through all three experiments and prints
results to stdout. There are no command-line arguments — edit the script
directly to change architectures, epochs, batch size, or the number of folds.

## Methodology

- **Preprocessing:** Features are standardized (`StandardScaler`) separately
  within each cross-validation fold — the scaler is fit on training data only
  and applied to the held-out fold, preventing data leakage.
- **Validation:** 10-fold CV (`KFold(n_splits=10, shuffle=True, random_state=42)`)
- **Training:** 100 epochs, batch size 16, Adam optimizer, MSE loss
- **Metric:** Mean Squared Error (MSE), averaged (and std-dev'd) across folds

## Results

### 1. Two Hidden Layer MLP (64 → 32)

| Metric | Value |
|---|---|
| Mean MSE | 11.39 |
| Std Dev | 5.51 |

### 2. Single Hidden Layer — Effect of Hidden Units

| Hidden Units | Mean MSE |
|---|---|
| 8   | 29.41 |
| 16  | 17.53 |
| 32  | 14.33 |
| 64  | 12.60 |
| 128 | 11.84 |

More units consistently reduced MSE; the biggest gains came between 16 and
64 units, with diminishing returns afterward.

### 3. Arbitrary Depth — Effect of Network Depth

| Architecture | Mean MSE |
|---|---|
| [64] | 12.75 |
| [64, 32] | 12.00 |
| [64, 32, 16] | 10.72 |
| [128, 64, 32, 16] | **9.34** |

Deeper networks consistently outperformed shallower ones, with
`[128, 64, 32, 16]` achieving the best result overall.

## Conclusions

- Per-fold standardization improved training stability and convergence.
- The baseline two-hidden-layer model already beat the typical benchmark
  MSE of ~20 by a wide margin.
- Increasing hidden units (single-layer case) and increasing depth
  (multi-layer case) both improved performance, with depth giving the best
  overall result: **MSE = 9.34** for `[128, 64, 32, 16]`.
- As with most MLP tuning, there's a tradeoff between capacity and
  overfitting/compute cost — the improvements here plateau rather than
  growing indefinitely with size.

## Environment

- Python 3.x
- TensorFlow (Keras API)
- scikit-learn
- NumPy
- Pandas
