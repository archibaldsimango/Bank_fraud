from flask import Flask,render_template,request
import pickle
import pandas as pd

model1 = pickle.load(open('xgboost_model.sav','rb'))
model2 = pickle.load(open('logistic_regr_model.sav','rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=["GET","POST"])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))

    if request.method == "POST":
        step= request.form['step']
        gender = request.form['gender']
        merchant = request.form['merchant']
        category = request.form['category']
        amount = request.form['amount']

        input_variables = pd.DataFrame([[step, gender, merchant,category, amount]],columns=['step', 'gender', 'merchant',
        'category', 'amount'])

        prediction1 = model1.predict(input_variables)
        prob1 = model1.predict_proba(input_variables)[0]
        factor1 = prob1[0] *100
        confidence_factor1 = str(round(factor1, 1)) +'%'
        if prediction1 == 0:
            result1 = 'No Fraud'
        else:
            result1 = 'Fraud'

        prediction2 = model2.predict(input_variables)
        prob2 = model2.predict_proba(input_variables)[0]
        factor2 = prob2[0] * 100
        confidence_factor2 = str(round(factor2, 1)) + '%'
        if prediction2 == 0:
            result2 = 'No Fraud'
        else:
            result2 = 'Fraud'

        return render_template('main.html',result1=result1,confidence_factor1=confidence_factor1,result2=result2,confidence_factor2=confidence_factor2)

    if __name__ == '__main__':
        app.debug = True
        app.run()
        