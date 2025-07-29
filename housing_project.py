from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the California Housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Show some info about the data
print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of the dataset:")
print(df.shape)  # (number of rows, number of columns)

print("\nAvailable columns/features:")
print(df.columns.tolist())

print("\nDataset description:")
print(california.DESCR)

import matplotlib.pyplot as plt

# Plot histograms for all features
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('MedHouseVal', axis=1)  # Features (everything except the price)
y = df['MedHouseVal']               # Target (price)

# Scale the features so they are all on a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('MedHouseVal', axis=1)  # All columns except target
y = df['MedHouseVal']               # The target column

# Scale features (important for ML models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)      # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)       # Mean Squared Error
r2 = r2_score(y_test, y_pred)                  # R^2 Score

print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")
import joblib

# Save the trained model
joblib.dump(model, 'linreg_model.joblib')

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler files saved!")
