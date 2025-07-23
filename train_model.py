import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
laptop_data = pd.read_csv('laptops.csv')

# Select features and target
features = [
    'Status', 'Brand', 'Model', 'CPU', 'RAM', 'Storage', 'Storage type', 'GPU', 'Screen', 'Touch'
]
target = 'Final Price'

X = laptop_data[features]
y = laptop_data[target]

# Identify categorical and numerical columns
categorical_cols = ['Status', 'Brand', 'Model', 'CPU', 'Storage type', 'GPU', 'Touch']
numerical_cols = ['RAM', 'Storage', 'Screen']

# Fill missing values
for col in categorical_cols:
    X[col] = X[col].fillna('Unknown')
for col in numerical_cols:
    X[col] = X[col].fillna(X[col].median())

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

print('Model trained and saved as model.pkl') 