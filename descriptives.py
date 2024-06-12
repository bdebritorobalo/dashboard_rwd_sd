from tableone import TableOne
import pandas as pd

# Load the dataset using pandas
file_path = '/home/bruno@mydre.org/dashboard/dashboard_rwd_sd/dataset/20240603_data_processed.csv'
data = pd.read_csv(file_path)

# Update these column names based on your dataset's actual columns
columns = ['main-procedure', 'procedure_duration', 'procedure_age', 'postop_status_sternum', 'ECC_duration', 'AoX_duration', 'DHCA_duration', 'ACP_duration']
categorical = ['main-procedure', 'postop_status_sternum','HER2 postop_status_sternum']
continuous = ['ECC_duration', 'AoX_duration', 'DHCA_duration', 'ACP_duration', 'procedure_age']

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
