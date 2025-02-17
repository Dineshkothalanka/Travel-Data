import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the dataset file exists
file_path = "Travel.csv"
if not os.path.exists(file_path):
    logger.error(f"Dataset file '{file_path}' not found")
    raise FileNotFoundError(f"Dataset file '{file_path}' not found. Please check the file path and try again.")


# Load dataset and perform initial exploration
df = pd.read_csv(file_path)
logger.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

# Display basic info
logger.info("\nDataset Info:")
df.info()

# Display descriptive statistics
logger.info("\nDescriptive Statistics:")
logger.info(df.describe())

# Check for missing values
logger.info("\nMissing Values:")
logger.info(df.isnull().sum())

# Drop CustomerID as it is not useful for prediction
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Handling missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numerical columns with median
num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical columns with most frequent value
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Encoding categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Plot correlation matrix using only numerical columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Scaling numerical features (excluding target variable)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Clean and validate target variable
logger.info("\nUnique values in ProdTaken before cleaning:")
logger.info(df['ProdTaken'].unique())

# Convert to integer and handle any unexpected values
df['ProdTaken'] = df['ProdTaken'].astype(int)
df['ProdTaken'] = df['ProdTaken'].apply(lambda x: 1 if x > 0 else 0)

logger.info("\nUnique values in ProdTaken after cleaning:")
logger.info(df['ProdTaken'].unique())

# Split dataset into training and testing sets
X = df.drop(columns=['ProdTaken'])
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    logger.info(f"\nEvaluation Metrics for {model_name}:")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

# Apply multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}

best_model = None
best_score = 0

for name, model in models.items():
    logger.info(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred_proba)
    else:
        score = model.score(X_test, y_test)
    
    if score > best_score:
        best_score = score
        best_model = name
    
    evaluate_model(y_test, y_pred, name)

logger.info(f"\nBest performing model: {best_model} with score: {best_score:.4f}")

# Save the processed dataset and model results
df.to_csv("Processed_Travel.csv", index=False)
logger.info("Data preprocessing and analysis complete. Processed data saved as 'Processed_Travel.csv'.")
