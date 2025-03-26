from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load trained model and encoder from the static directory
try:
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    model_path = os.path.join(static_dir, 'model.pkl')
    with open(model_path, "rb") as f:
        model, encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    # In production, you might want to handle this more gracefully
    raise

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    # Use environment variable for port, defaulting to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
