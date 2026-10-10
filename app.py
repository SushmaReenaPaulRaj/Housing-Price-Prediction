from flask import Flask, request, render_template
import pandas as pd
from xgboost import Booster, DMatrix

app = Flask(__name__)

# Load the model using Booster
xgb_model = Booster(model_file='xgb_model.txt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = {
            'area': [float(request.form['area'])],
            'bedrooms': [int(request.form['bedrooms'])],
            'bathrooms': [int(request.form['bathrooms'])],
            'stories': [int(request.form['stories'])],
            'mainroad': [int(request.form['mainroad'])],
            'guestroom': [int(request.form['guestroom'])],
            'airconditioning': [int(request.form['airconditioning'])],
            'parking': [int(request.form['parking'])],
            'prefarea': [int(request.form['prefarea'])],
            'furnishingstatus': [int(request.form['furnishingstatus'])]
        }

        # Convert input data to a DataFrame
        test_input = pd.DataFrame(input_data)

        # Convert DataFrame to DMatrix
        dmatrix = DMatrix(test_input)

        # Predict using the loaded model
        predicted_price = xgb_model.predict(dmatrix)[0]

        # Render the result template with the predicted price
        return render_template('result.html', predicted_price=predicted_price)

    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
