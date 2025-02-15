import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the cleaned dataset
df = pd.read_csv(r'D:\Housing\data\c1.csv')  # Ensure this file exists in the correct path

# Define features and target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Convert categorical features to category dtype for LightGBM
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_features:
    X[col] = X[col].astype("category")



# Split dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Define the LightGBM model
model = lgb.LGBMRegressor()
"""
# Define parameter grid for RandomizedSearchCV
param_grid = {
    'num_leaves': [5,7,10,15],
    #'learning_rate': [0.005, 0.01, 0.05],
    'n_estimators': [100, 500, 1000],
    'max_depth': [3,5,7],
    'min_child_samples': [30, 50, 70],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha':[0.5, 1.0, 2.0],
    'reg_lambda':[0.05, 0.1, 0.5]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_grid, n_iter=20, cv=5, scoring='neg_root_mean_squared_error',
                                   verbose=1, random_state=42, n_jobs=-1) """

# Define the LightGBM model with default parameters
model = lgb.LGBMRegressor(force_row_wise=True)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

# Print cross-validation results
print(f"Cross-validation RMSE scores: {-cv_scores}")
print(f"Mean RMSE: {-cv_scores.mean()}")

# Fit the model on the full training data
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_val, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Save the model
joblib.dump(model, "lightgbm_model.pkl")
print("Model saved successfully!")
