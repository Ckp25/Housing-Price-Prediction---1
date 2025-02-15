import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import skew
import numpy as np

# Load the cleaned dataset
df = pd.read_csv(r'D:\Housing\data\c1.csv')  # Ensure this file exists in the correct path
test_df = pd.read_csv(r'D:\Housing\data\c2.csv')  # Ensure this file exists

# Define features and target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Convert categorical features to category dtype for LightGBM
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_features:
    X[col] = X[col].astype("category")
    test_df[col] = test_df[col].astype("category")


# Split dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LightGBM model
model = lgb.LGBMRegressor()

# Define parameter grid for RandomizedSearchCV
param_grid = {
    'num_leaves': [20, 31, 40, 50],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 500, 1000],
    'max_depth': [-1, 5, 10, 15],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, scoring='neg_root_mean_squared_error',
                                   verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train, categorical_feature=categorical_features)

# Get best model
best_model = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_val, y_pred)

print(f"Best Params: {random_search.best_params_}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Save the best model
joblib.dump(best_model, "lbm_randomCV_model.pkl")
print("Model saved successfully!")
