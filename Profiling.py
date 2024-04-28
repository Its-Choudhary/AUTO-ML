import pandas as pd
from pandas_profiling import ProfileReport

# Load your dataset (replace 'your_dataset.csv' with your actual file path)
df = pd.read_csv('Recived_csv.csv')

# Generate the profile report
profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)

# Save the report to a file
profile.to_file('your_report.html')
