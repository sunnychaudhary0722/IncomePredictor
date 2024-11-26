from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('canada_per_capita_income.csv')

# Create the regression model
reg = LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input year from the form
    year = int(request.form['year'])
    
    # Predict the per capita income for the given year
    predicted_income = reg.predict([[year]])[0]
    
    return jsonify({'predicted_income': predicted_income})

if __name__ == '__main__':
    app.run(debug=True)
