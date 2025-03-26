from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and encoder
model, encoder = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    square_feet = float(request.form['square_feet'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    year_built = int(request.form['year_built'])
    neighborhood = request.form['neighborhood']

    # Encode the neighborhood
    neighborhood_encoded = encoder.transform([[neighborhood]])
    neighborhood_df = np.array(neighborhood_encoded).flatten()

    # Create input array for prediction
    input_data = np.array([[square_feet, bedrooms, bathrooms, year_built] + list(neighborhood_df)])
    
    # Predict price
    prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction_text=f"Estimated Price: ${prediction:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)
