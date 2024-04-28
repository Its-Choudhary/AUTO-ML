import pandas as pd
from pycaret.classification import *

# Load the dataset
df = pd.read_csv('Recived_csv.csv')

# Select the column you want to use as the target variable
selected_column = 'Sex'

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
