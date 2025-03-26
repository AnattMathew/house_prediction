import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)

# Load dataset from the static directory
data_path = os.path.join(static_dir, 'housing_price_dataset.csv')
df = pd.read_csv(data_path)

# Handle missing values
df.dropna(inplace=True)

# One-Hot Encode 'Neighborhood'
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_neighborhoods = encoder.fit_transform(df[['Neighborhood']])

# Convert to DataFrame and merge
encoded_df = pd.DataFrame(encoded_neighborhoods, columns=encoder.get_feature_names_out(['Neighborhood']))
df = pd.concat([df.drop(columns=['Neighborhood']), encoded_df], axis=1)

# Define features (X) and target (y)
X = df.drop(columns=['Price'])
y = df['Price']

# Split into training and testing sets (80%-20% split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model in the static directory
model_path = os.path.join(static_dir, 'model.pkl')
with open(model_path, "wb") as f:
    pickle.dump((model, encoder), f)

print(f"Model trained and saved as {model_path}")
