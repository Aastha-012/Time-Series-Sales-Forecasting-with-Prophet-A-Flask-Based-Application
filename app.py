from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
from datetime import datetime

app = Flask(__name__)

# Load historical data
data = pd.read_csv('sales_data.csv')

# Preprocess the data
data['ds'] = pd.to_datetime(data['Date'])
data['y'] = data['Sales']
data = data[['ds', 'y']]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to generate forecasts
@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get the forecast period (number of days) from the form
        period = int(request.form['forecast_period'])

        # Initialize and train the Prophet model
        model = Prophet()
        model.fit(data)

        # Create a future dataframe and generate predictions
        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)

        # Filter only future data for the forecasted period
        forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)

        # Convert the results to a list of dictionaries for rendering
        forecast_results = forecast_results.to_dict(orient='records')

        return render_template('index.html', forecast_results=forecast_results)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
