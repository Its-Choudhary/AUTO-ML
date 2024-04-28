from flask import Flask, request, jsonify, send_file
import pandas as pd
from pycaret.classification import *
from pandas_profiling import ProfileReport

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
    #storing it in a variable or database
    print("Received target column value:",target_column_value)
    return run_pycaret_classification("Recived_csv.csv", target_column_value)


if __name__ == '__main__':
    app.run(debug=True)




#---------------------------------------ML.py running--------------------------------##

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
    return "Runned Sucessfully"




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
    profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

    # Save the report to a file
    profile.to_file("your_report.html")










