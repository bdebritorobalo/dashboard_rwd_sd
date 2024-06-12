from tableone import TableOne
import pandas as pd

# Load the dataset using pandas
file_path = '/workspaces/dashboard_rwd_sd/dataset/breast_cancer_survival.csv'
data = pd.read_csv(file_path)

# Update these column names based on your dataset's actual columns
columns = ['Age', 'Gender', 'Protein1', 'HER2 status', 'Patient_Status']
categorical = ['Gender', 'Patient_Status']
continuous = ['Age', 'Protein1']

# Check if specified columns exist in the dataframe
missing_columns = [col for col in columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Check for non-numeric values in continuous columns
non_numeric_continuous = [col for col in continuous if not pd.api.types.is_numeric_dtype(data[col])]
if non_numeric_continuous:
    raise ValueError(f"Non-numeric values found in continuous columns: {non_numeric_continuous}")

# Groupby and nonnormal columns
groupby = 'death'
nonnormal = ['Age']
rename = {'death': 'mortality'}

# Create the TableOne object
mytable = TableOne(data)

# Print the descriptive statistics table
print(mytable.tabulate(tablefmt="fancy_grid"))
