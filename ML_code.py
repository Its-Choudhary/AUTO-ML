# Import necessary libraries
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *

# Load the dataset
data = pd.read_csv('Recived_csv.csv')

# Initialize the setup and detect the problem type
if data[target_column].dtype == 'object':
    s = setup(data, target=target_column, silent=True)
    problem_type = 'classification'
    compare_models_func = compare_models
    tune_model_func = tune_model
    predict_model_func = predict_model
else:
    s = setup(data, target=target_column, silent=True)
    problem_type = 'regression'
    compare_models_func = compare_models_reg
    tune_model_func = tune_model_reg
    predict_model_func = predict_model_reg

# Explore and preprocess the data
explore_data()
prep_data()

# Compare different models
models = compare_models_func(fold=5)

# Tune the hyperparameters of the best model
tuned_model = tune_model_func(models[0])

# Log the experiment
exp_name = log_experiment(tuned_model, 'experiment_name')

# Create a pipeline
pipeline = create_pipeline(tuned_model)

# Make predictions
predictions = predict_model_func(pipeline, data=new_data)

# Save the best model
save_model(tuned_model, 'Model/best_model')

# Save the model as a pickle file
save_model(tuned_model, 'Model/best_model_pickle.pkl')

# Create a file for the plots
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_model(tuned_model, plot='residuals')
plt.savefig('Plots/residuals.png')

plt.figure(figsize=(12, 8))
plot_model(tuned_model, plot='error')
plt.savefig('Plots/error.png')

# Add more plots as needed