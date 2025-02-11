# Lasso Feature Selection and Polynomial Regression on Wine Quality Dataset

## Overview
This project applies Lasso regression for feature selection before performing Polynomial Regression to predict wine quality. The model is implemented in Python using `scikit-learn` and `pandas` within a Jupyter Notebook.

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

3. **Feature Scaling:**
   - Standardize features using `StandardScaler`.

4. **Lasso Feature Selection:**
   - Apply `Lasso` regression (`alpha=0.1`) to select important features.
   - Remove features with zero coefficients.

5. **Polynomial Feature Expansion:**
   - Use `PolynomialFeatures(degree=2)` to expand feature space after feature selection.

6. **Data Splitting:**
   - Split dataset into training (70%) and testing (30%) sets using `train_test_split`.

7. **Feature Scaling (After Expansion):**
   - Standardize the newly transformed features using `StandardScaler`.

8. **Model Training:**
   - Train a `LinearRegression` model on the selected and transformed data.

9. **Model Evaluation:**
   - Predict test set values.
   - Evaluate using Mean Squared Error (MSE) and R-squared (R2) score.

## Outputs
- **Lasso Coefficients:** Shows the number of selected features.
- **Model Coefficients:** Displayed after training.
- **Mean Squared Error:** Measures prediction accuracy.
- **R2 Score:** Evaluates how well the model explains variance in the data.

## Running the Notebook
To execute the notebook, open Jupyter Notebook and run the provided Python script step by step.

```bash
jupyter notebook
```

