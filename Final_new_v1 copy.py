from flask import Flask, request, jsonify, send_file
import pandas as pd
from pycaret.classification import *

from pycaret.regression import *

from pandas_profiling import ProfileReport

from typing import Optional, List, Union, Any, Dict



app = Flask(__name__)

users = {
    'test@gmail.com': 'test'
}

##------------------------------------- Login  API------------------------------------------##

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    print("Received email:", email)
    print("Received password:", password)

    if email in users and users[email] == password:
        return jsonify({'message': 'Login successful', 'token': 'your_generated_token_here'})
    else:
        return jsonify({'message': 'Invalid username or password '}), 401


##------------------------------------ Sign Up  API ------------------------------------------##

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if email in users:
        return jsonify({'message': 'Email already exists, please use a different email.'}), 400

    # Add the new user to the dictionary
    users[email] = password
    print("User Dict-:",users)
    return jsonify({'message': 'Sign up successful'}), 200



##----------------------------------- CSV file Reciver ---------------------------------------##
@app.route('/upload', methods=['POST'])
def upload_file():
    print("Request data:", request.data)
    
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    file.save('Recived_csv.csv')             # Save received csv file on local machine with fix name "Recived_csv.csv"
    return 'File uploaded successfully'

##----------------------------------- Sending Html FIle to server  ------------------------------##
@app.route('/get_html_file', methods=['GET'])
def get_html_file():
    generate_profile_report('Recived_csv.csv')
    return send_file('your_report.html', mimetype='text/html')




##------------------------------final ML model settings  csv file send to Retool-----------------------------------------###


@app.route('/ML_Final_CSV_send_To_Retool', methods=['GET'])
def ML_Final_CSV_send_To_Retool():
    df = pd.read_csv('setup_dataframe.csv')
    csv_data = df.to_csv(index=False)
    return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=setup_dataframe.csv'}






##---------------------------------Final ML model list+ Perfromance csv file send to retool ------------#


@app.route('/ML_Final_CSV_Model_list_send_To_Retool', methods=['GET'])
def ML_Final_CSV_Model_list_send_To_Retool():
    df = pd.read_csv('compare_dataframe.csv')
    csv_data = df.to_csv(index=False)
    return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=setup_dataframe.csv'}












##------------------------------Send CSV file for Target Column---------------------###

@app.route('/Rcv_CSV_initial_target_column', methods=['GET'])
def Rcv_CSV_initial_target_column():
    df = pd.read_csv('Recived_csv.csv')
    csv_data = df.to_csv(index=False)
    return csv_data, 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=setup_dataframe.csv'}




##-------------------------------Recive Target Componment Value-------------------------##

# Route to retrieve the stored value


@app.route('/store', methods=['POST'])
def store_value():


    # Handle the POST request here
    data1 = request.json  # Access the JSON data sent in the request body
    target_column_value = data1.get("Target_value")  # Get the value of the target column from the request
    Model_type_value =  data1.get("Model_type") # type of model  (Regression or Classification )
    #storing it in a variable or database
    print("Received target column value:",target_column_value)
    print("Received Model type column  value:",Model_type_value)

    if Model_type_value == 0:
        return run_pycaret_classification('Recived_csv.csv', target_column_value)
    elif Model_type_value == 1:
        return run_pycaret_regression('Recived_csv.csv', target_column_value)
    else:
        return jsonify({'message': 'Invalid task type'}), 400


#---------------------------------------ML.py running for Classificatin --------------------------------##

def run_pycaret_classification(csv_file, target_column):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Select the column you want to use as the target variable
    selected_column = target_column

    # Setup PyCaret environment for classification
    setup(data=df, target=selected_column)

    # Pull the setup dataframe
    setup_df = pull()
    dataframe_setup = setup_df

    # Save setup dataframe to CSV
    dataframe_setup.to_csv('setup_dataframe.csv', index=False)

    # Compare models
    best_model = compare_models()

    # Pull the compare dataframe
    compare_df = pull()
    dataframe_compare = compare_df

    # Save compare dataframe to CSV
    dataframe_compare.to_csv('compare_dataframe.csv', index=False)
    return "Classification task Run Sucessfully"

##--------------------------------ML code for Regression---------------------------##


def run_pycaret_regression(csv_file, target_column):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Setup PyCaret environment for regression
    setup(data=df, target=target_column)

    setup_df = pull()
    dataframe_setup = setup_df

    # Save setup dataframe to CSV
    dataframe_setup.to_csv('setup_dataframe.csv', index=False)


    # Compare models for regression
    best_model = compare_models(fold=5, sort='R2')

    # Pull the compare dataframe
    compare_df = pull()
    dataframe_compare = compare_df

    # Save compare dataframe to CSV
    dataframe_compare.to_csv('compare_dataframe.csv', index=False)

    return "Regression task completed successfully"


##-----------------------------Pandas Profiling of Dataset-----------------------##

def generate_profile_report(csv_file_path):
    """
    Generate a profile report for the dataset and save it to a file.

    Parameters:
        csv_file_path (str): The file path to the CSV dataset.
        report_file_path (str): The file path to save the generated report.
    """
    # Load the dataset
    df = pd.read_csv(csv_file_path)

    # Generate the profile report
    profile = ProfileReport(df, title=' Profiling Report', explorative=True)

    # Save the report to a file
    profile.to_file("your_report.html")

if __name__ == '__main__':
    app.run(debug=True)






def compare_models(
    include: Optional[List[Union[str, Any]]] = None,
    exclude: Optional[List[str]] = None,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "R2",
    n_select: int = 1,
    budget_time: Optional[float] = None,
    turbo: bool = True,
    errors: str = "ignore",
    fit_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    experiment_custom_tags: Optional[Dict[str, Any]] = None,
    engine: Optional[Dict[str, str]] = None,
    verbose: bool = True,
):
    pass

