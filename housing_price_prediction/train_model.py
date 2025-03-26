import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("housing_price_dataset.csv")

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

# Save the trained model
pickle.dump((model, encoder), open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")
