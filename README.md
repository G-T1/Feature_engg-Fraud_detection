# Feature_engg-Fraud_detection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample dataset creation
df = pd.read_csv(r'C:\Users\Gnan Tejas D\Downloads\student_scores.csv')

# ✅ Feature Engineering: Create a new total score column
df['Total_Score'] = df['Math_Score'] + df['Science_Score'] + df['English_Score']

# Features and target
X = df[['Math_Score', 'Science_Score', 'English_Score', 'Total_Score']]
y = df['Pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Model and hyperparameter tuning
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
