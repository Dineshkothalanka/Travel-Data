# Travel Dataset Analysis Project

## Overview
This project focuses on analyzing customer travel data to predict product purchase decisions. It includes a comprehensive data preprocessing pipeline and machine learning models implemented in Python using scikit-learn.

## Dataset
The dataset contains 4888 records with 20 features including:
- Customer demographics (Age, Gender, Marital Status)
- Travel preferences (CityTier, PreferredPropertyStar)
- Interaction details (DurationOfPitch, NumberOfFollowups)
- Target variable: ProdTaken (whether the customer purchased the product)

## Features
- Data preprocessing: Missing value imputation, categorical encoding, feature scaling
- Machine learning: Logistic Regression and Random Forest classifiers
- Visualization: Correlation matrices, confusion matrices
- Robust logging and error handling

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Dineshkothalanka/Travel-Data.git
cd Travel-Data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the analysis pipeline:
```bash
python pds.py
```

This will:
- Preprocess the data
- Train and evaluate models
- Generate visualizations
- Save processed data and results

## Results
The project generates:
- Correlation matrix (correlation_matrix.png)
- Model evaluation metrics
- Confusion matrices for each model
- Processed dataset (Processed_Travel.csv)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
