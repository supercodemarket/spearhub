from flask import Flask, request, jsonify

import pandas as pd

app = Flask(__name__)

# Placeholder for the trained model
model = None

# Endpoint for uploading the dataset
@app.route('/upload', methods=['POST'])
def upload_data():
    file = request.files['file']
    df = pd.read_csv('defect_dataset.csv')
    global model
    
    # Preprocessing
    X = df[['Temperature', 'Run_Time']]
    y = df['Downtime_Flag']
    
    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    return jsonify({"message": "Data uploaded and model trained successfully!"})

# Endpoint for training the model
@app.route('/train', methods=['POST'])
def train_model():
    if model is None:
        return jsonify({"error": "No model to train. Upload data first."}), 400
    
    # Assume data has been uploaded in the /upload route
    return jsonify({"message": "Model is already trained."})

# Endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"This is your prediction"}), 400
    
    data = request.get_json()
    temperature = data['Temperature']
    run_time = data['Run_Time']
    
    # Predict downtime flag
    prediction = model.predict([[temperature, run_time]])
    confidence = model.predict_proba([[temperature, run_time]])[0][prediction[0]]
    
    return jsonify({
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": confidence
    })

if __name__ == '__main__':
    app.run(port=5001)
