from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests

# Load the trained fraud detection model and scaler
fraud_model = joblib.load(r"C:\Users\Rosh05\Desktop\bot\models\fraud_model.pkl")
scaler = joblib.load(r"C:\Users\Rosh05\Desktop\bot\models\scaler.pkl")

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json()
    intent_name = req['queryResult']['intent']['displayName']

    # Handle different intents
    if intent_name == "Upload CSV Intent":
        return handle_csv_upload(req)
    elif intent_name == "Results Query Intent":
        return handle_results_query(req)
    elif intent_name == "Help Intent":
        return jsonify({"fulfillmentText": "You can upload a CSV file, and I will detect fraud in it."})
    else:
        return jsonify({"fulfillmentText": "Sorry, I don't understand that."})

def handle_csv_upload(req):
    try:
        # Get the file URL from the request (Dialogflow sends it)
        file_url = req['queryResult']['parameters']['file_url']
        file = requests.get(file_url).content

        # Load CSV from the URL content
        data = pd.read_csv(pd.compat.StringIO(file.decode('utf-8')))

        # Preprocess the data: drop 'is_fraud' column, as it's the target
        features = data.drop('is_fraud', axis=1)

        # Scale features if you have used scaling during training (if applicable)
        X_scaled = scaler.transform(features)

        # Make fraud predictions
        fraud_predictions = fraud_model.predict(X_scaled)

        # Add the fraud prediction to the data
        data['Fraud_Prediction'] = fraud_predictions

        # Save the predictions (optional)
        data.to_csv(r"C:\Users\Rosh05\Desktop\bot\predictions.csv", index=False)

        return jsonify({"fulfillmentText": "Your file has been processed. Fraud predictions are ready!"})

    except Exception as e:
        return jsonify({"fulfillmentText": f"An error occurred: {str(e)}"})

def handle_results_query(req):
    try:
        # Load previously saved predictions
        data = pd.read_csv(r"C:\Users\Rosh05\Desktop\bot\predictions.csv")

        # Summarize results
        fraud_count = data['Fraud_Prediction'].sum()
        total = len(data)

        return jsonify({"fulfillmentText": f"I found {fraud_count} fraudulent transactions out of {total} total transactions."})
    
    except Exception as e:
        return jsonify({"fulfillmentText": f"Could not fetch results: {str(e)}"})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
