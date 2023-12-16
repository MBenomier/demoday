from flask import Flask, render_template, request
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from model import model, base, create_period 
print(model.get_booster().feature_names)
app = Flask(__name__)


features = ['hour', 'minute', 'day', 'month', 'year', 'dayofweek', 'dayofyear']#[ 'hour','minute', 'day', 'month', 'year','dayofweek', 'dayofyear']

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/', methods=['GET','POST'])
def predict():
    start_date = request.form['fd']
    end_date = request.form['td']
    df = create_period(start_date,end_date) 
    fut = model.predict(df[features])
    future = pd.DataFrame(data={'Predictions':fut}, index = df.index)
    plt.switch_backend('Agg') 
    plt.figure(figsize=(20,5))
    plt.plot(future['Predictions'])
    plt.savefig('./static/output.png')

    return render_template('index.html') 


if __name__ == "__main__":
    app.run(debug=True)

