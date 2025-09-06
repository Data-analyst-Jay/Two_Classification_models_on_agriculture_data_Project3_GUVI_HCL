# Project 3 — Model

Short overview
- Two Jupyter notebooks for agricultural ML tasks:
  - soil_fertility_prediction.ipynb — binary soil fertility classification using Logistic Regression (RobustScaler, L1 regularization, GridSearchCV).
  - crop_recommendation.ipynb — crop classification using Decision Tree and Random Forest (RobustScaler, GridSearchCV).

Repository structure
- soil_fertility_prediction.ipynb — EDA, preprocessing, model training, cross-validation, hyperparameter search.
- crop_recommendation.ipynb — EDA, scaling, baseline Decision Tree, RandomForest, GridSearchCV tuning.
- requirement.txt — Python package dependencies.

Requirements
- Python 3.8+ recommended
- Packages listed in requirement.txt:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

Setup (Windows)
1. Create and activate a virtual environment:
   - python -m venv venv
   - venv\Scripts\activate
2. Install dependencies:
   - pip install -r requirement.txt
3. Start Jupyter:
   - jupyter notebook
4. Open and run notebooks in the Model folder.

Data
- Place the dataset CSV files next to the notebooks:
  - Soil_fertility_prediction.csv (used by soil_fertility_prediction.ipynb)
  - Crop_Prediction_Dataset.csv (used by crop_recommendation.ipynb)
- Included in this folder. Ensure correct filenames and columns expected by each notebook.

Key implementation notes
- Both notebooks use RobustScaler to handle skewed features.
- soil_fertility_prediction:
  - Re-maps Output label '2' to '1' to form a binary target.
  - Uses LogisticRegression with L1 penalty and GridSearchCV over C values.
  - Reports CV mean accuracy and test accuracy.
- crop_recommendation:
  - Drops CROP_PRICE and STATE columns before training.
  - Uses DecisionTree as baseline and RandomForest for improved performance.
  - Uses GridSearchCV to tune RandomForest hyperparameters and evaluate best estimator.

Reproducibility tips
- Fix random_state where needed (notebooks already use random_state=42 in key places).
- Use the same scaler fit on training data for test and production transforms.
- Save best estimators (joblib or pickle) after GridSearchCV for deployment.

Suggestions / Next steps
- Add model serialization (joblib.dump) and a small inference script or API.
- Add unit tests for preprocessing and a requirements-dev.txt for linting/testing tools.
- Include sample input CSVs or a data README describing column names and types.