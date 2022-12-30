from flask import Flask, request, jsonify
import main

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the request
    data = request.get_json()
    definition = data['definition']

    # Make a prediction using the main script
    prediction = main.predict(definition)

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()
