from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np  # Added import for saving background data

# Load the California Housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Separate features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Save SHAP background data: 100 samples from scaled training set
np.save('background_scaled.npy', X_train[:100])

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, 'linreg_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model, scaler, and background data saved!")
