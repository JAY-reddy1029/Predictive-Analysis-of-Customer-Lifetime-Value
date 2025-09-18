# =====================================================
# Predictive Analysis of Customer Lifetime Value (CLV)
# Dataset: FinTech Customer LTV Dataset (Kaggle)
# =====================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Predictive_Analysis_of_Customer_Lifetime_Value\digital_wallet_ltv_dataset.csv")  # Change to your file path

# Step 3: Quick Data Overview
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# Step 4: Handle Missing Values
df = df.dropna()

# Step 5: Encode Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Step 6: Define Features and Target
# Drop Customer_ID (identifier, not useful for prediction)
X = df.drop(["Customer_ID", "LTV"], axis=1)
y = df["LTV"]

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Model Training with Hyperparameter Tuning (Random Forest)
rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)

best_rf = grid_rf.best_estimator_

# Step 10: Model Evaluation
y_pred = best_rf.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n✅ Best Random Forest Parameters:", grid_rf.best_params_)
print("📊 R² Score:", round(r2, 4))
print("📉 MAE:", round(mae, 2))
print("📉 RMSE:", round(rmse, 2))

# Step 11: Feature Importance Visualization
importances = best_rf.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in LTV Prediction")
plt.show()


import pickle

# Save the trained model
with open("rf_ltv_model.pkl", "wb") as file:
    pickle.dump(best_rf, file)

# Save the scaler as well (needed to scale new input data)
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("✅ Model and scaler saved successfully!")

import pickle

# Save the trained scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("✅ Scaler saved successfully!")


