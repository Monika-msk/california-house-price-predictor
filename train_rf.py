from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np

# 1. Load and prepare data
california = fetch_california_housing(as_frame=True)
df = california.frame
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. GridSearchCV for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='neg_mean_absolute_error',
    verbose=2
)

# 5. Fit GridSearchCV
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# 6. Save artifacts for app.py
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(best_rf_model, 'rf_model.joblib')
np.save('rf_importances.npy', best_rf_model.feature_importances_)

# 7. Print best parameters and status
print("Best RF Parameters:", grid_search.best_params_)
print("GridSearchCV training complete and model saved!")
