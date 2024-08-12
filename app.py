from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

try:
    # Load the scaler and model
    with open('model/standarScaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    with open('model/modelForPrediction.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET','POST'])
def classify():
    try:
        # Extract all required features from the form
        gill_spacing = float(request.form.get('gill_spacing', 0))
        gill_size = float(request.form.get('gill_size', 0))
        gill_color = float(request.form.get('gill_color', 0))
        stalk_shape = float(request.form.get('stalk_shape', 0))
        stalk_root = float(request.form.get('stalk_root', 0))
        stalk_surface_above_ring = float(request.form.get('stalk_surface_above_ring', 0))
        stalk_surface_below_ring = float(request.form.get('stalk_surface_below_ring', 0))
        stalk_color_above_ring = float(request.form.get('stalk_color_above_ring', 0))
        stalk_color_below_ring = float(request.form.get('stalk_color_below_ring', 0))
        veil_type = float(request.form.get('veil_type', 0))
        veil_color = float(request.form.get('veil_color', 0))
        ring_number = float(request.form.get('ring_number', 0))
        ring_type = float(request.form.get('ring_type', 0))
        spore_print_color = float(request.form.get('spore_print_color', 0))
        population = float(request.form.get('population', 0))
        habitat = float(request.form.get('habitat', 0))

        # Combine features into a numpy array
        features = np.array([[gill_spacing, gill_size, gill_color, stalk_shape, stalk_root, 
                              stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, 
                              stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type, 
                              spore_print_color, population, habitat]])

        # Scale the features using the loaded StandardScaler
        scaled_features = scaler.transform(features)

        # Predict the class using the loaded model
        predicted_class = model.predict(scaled_features)[0]

        # Convert the predicted class to a human-readable format
        class_names = ['Edible', 'Poisonous']  # Adjust based on your model's output
        result = class_names[predicted_class]

        return render_template('result.html', prediction=result)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=False)
