from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            gender= int(request.form['gender'])
            age= int(request.form['age'])
            annual_salary = float(request.form['annual_salary'])
            credit_card_debt = float(request.form['credit_card_debt'])
            net_worth = float(request.form['net_worth'])
            pred_args = [gender, age, annual_salary, credit_card_debt, net_worth]
            pred_arr = np.array(pred_args)
            preds = pred_arr.reshape(1,-1)
            model = open("linear_regression_model.pkl","rb")
            lr_model = joblib.load(model)
            model_prediction = lr_model.predict(preds)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return 'Please Enter valid values'
    return render_template('predict.html', prediction = model_prediction)

if __name__ == '__main__':
    app.run()