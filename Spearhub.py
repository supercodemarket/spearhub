from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

app = Flask(__name__)

# Global variables to hold dataset and model
dataset = None
model = None

@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        dataset = pd.read_csv(file)
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/train', methods=['POST'])
def train_model():
    global model, dataset
    if dataset is None:
        return jsonify({"error": "No data to train the model"}), 400

    try:
        # Assuming columns "Temperature", "Run_Time", and "Downtime_Flag"
        X = dataset[['Temperature', 'Run_Time']]
        y = dataset['Downtime_Flag']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Make predictions and calculate performance metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return jsonify({
            "accuracy": accuracy,
            "f1_score": f1
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    try:
        data = request.json
        temperature = data['Temperature']
        run_time = data['Run_Time']

        prediction = model.predict([[temperature, run_time]])
        confidence = model.predict_proba([[temperature, run_time]])[0][prediction[0]]

        return jsonify({
            "Downtime": "Yes" if prediction[0] == 1 else "No",
            "Confidence": confidence
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
