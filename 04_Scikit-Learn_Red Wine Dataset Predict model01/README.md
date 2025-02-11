# Polynomial Regression on Wine Quality Dataset

## Overview
This project applies Polynomial Regression to predict wine quality using the `Wine Quality` dataset from the UCI Machine Learning Repository. The model is implemented in Python using `scikit-learn` and `pandas` within a Jupyter Notebook.

## Dataset
- **Source:** [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **File:** `winequality-red.csv`
- **Features:** Various physicochemical properties of wine
- **Target:** `quality` (wine quality score)

## Requirements
- Python 3.x
- Jupyter Notebook
- Required libraries:
  ```bash
  pip install numpy pandas scikit-learn
  ```

## Steps
1. **Load Dataset:**
   - Read `winequality-red.csv` using `pandas`.
   - Check for missing values.

2. **Define Features and Target:**
   - `X`: Features (all columns except `quality`).
   - `y`: Target variable (`quality`).

3. **Polynomial Feature Expansion:**
   - Use `PolynomialFeatures(degree=2)` to expand feature space.

4. **Data Splitting:**
   - Split dataset into training (70%) and testing (30%) sets using `train_test_split`.

5. **Feature Scaling:**
   - Standardize training and testing features using `StandardScaler`.

6. **Model Training:**
   - Train a `LinearRegression` model on the transformed data.

7. **Model Evaluation:**
   - Predict test set values.
   - Evaluate using Mean Squared Error (MSE) and R-squared (R2) score.

## Outputs
- **Model Coefficients:** Displayed after training.
- **Mean Squared Error:** Measures prediction accuracy.
- **R2 Score:** Evaluates how well the model explains variance in the data.

## Running the Notebook
To execute the notebook, open Jupyter Notebook and run the provided Python script step by step.

```bash
jupyter notebook
```

