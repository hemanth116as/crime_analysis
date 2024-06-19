# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('cleandata.csv')

# Preprocess the dataset (e.g., encoding categorical variables, handling missing values, etc.)
# ...

# Load the Machine Learning model
filename = 'svm_classifier.pkl'
model = pickle.load(open('svm_classifier.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Update the feature names to match the new dataset
        DR_NO = int(request.form.get('DR_NO'))
        Date_Rptd = request.form.get('Date_Rptd')
        DATE_OCC = request.form.get('DATE_OCC')
        TIME_OCC = request.form.get('TIME_OCC')
        AREA = request.form.get('AREA')
        AREA_NAME = request.form.get('AREA_NAME')
        Rpt_Dist_No = int(request.form.get('Rpt_Dist_No'))
        Part_1_2 = request.form.get('Part_1-2')
        Crm_Cd = request.form.get('Crm_Cd')
        Crm_Cd_Desc = request.form.get('Crm_Cd_Desc')
        Mocodes = request.form.get('Mocodes')
        Vict_Age = int(request.form.get('Vict_Age'))
        Vict_Sex = request.form.get('Vict_Sex')
        Vict_Descent = request.form.get('Vict_Descent')
        Premis_Cd = request.form.get('Premis_Cd')
        Premis_Desc = request.form.get('Premis_Desc')
        Weapon_Used_Cd = request.form.get('Weapon_Used_Cd')
        Weapon_Desc = request.form.get('Weapon_Desc')
        Status = request.form.get('Status')
        Status_Desc = request.form.get('Status_Desc')
        Crm_Cd_1 = request.form.get('Crm_Cd_1')
        LOCATION = request.form.get('LOCATION')
        Cross_Street = request.form.get('Cross_Street')
        LAT = request.form.get('LAT')
        LON = request.form.get('LON')
        Day_of_Week = request.form.get('Day_of_Week')
        Age_Group = request.form.get('Age_Group')
        Month = request.form.get('Month')

        # Preprocess the input data (e.g., encoding categorical variables, handling missing values, etc.)
        # ...

        # Create a numpy array with the input data
        data = np.array([[DR_NO, Date_Rptd, DATE_OCC, TIME_OCC, AREA, AREA_NAME, Rpt_Dist_No, Part_1_2, Crm_Cd, Crm_Cd_Desc, Mocodes, Vict_Age, Vict_Sex, Vict_Descent, Premis_Cd, Premis_Desc, Weapon_Used_Cd, Weapon_Desc, Status, Status_Desc, Crm_Cd_1, LOCATION, Cross_Street, LAT, LON, Day_of_Week, Age_Group, Month]])

        # Make a prediction using the model
        my_prediction = model.predict(data)

        # Return the prediction
        return render_template('results.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run()