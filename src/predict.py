import pandas as pd
import joblib

# Load the trained model
model = joblib.load('lbm_randomCV_model.pkl')

# Load the test dataset
test_df = pd.read_csv(r'D:\Housing\data\test.csv')  # Ensure this file exists

# Preserve the ID column for submission format
ids = test_df["Id"]

# Drop ID column and ensure same preprocessing as training
test = pd.read_csv(r'D:\Housing\data\c2.csv')

# Convert categorical features to category dtype
categorical_features = test.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_features:
    test[col] = test[col].astype("category")

# Predict on test data
y_pred = model.predict(test)

# Create submission file
submission = pd.DataFrame({"Id": ids, "SalePrice": y_pred})
submission.to_csv("submission03.csv", index=False)

print("Prediction completed. Submission file saved!")
