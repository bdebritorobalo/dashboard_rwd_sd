# File path: dashboard_analysis.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import socket
import os
import datetime
from flask import send_file
from dash.exceptions import PreventUpdate
from scipy.stats import ks_2samp, ttest_ind, chi2_contingency
from ctgan import CTGAN
import base64
import io

# Step 1: Load and Preprocess Data

def load_data(url):
    """Load data from a URL."""
    try:
        df = pd.read_csv(url)
        if df.empty:
            raise ValueError("The dataset is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from {url}: {e}")

def load_data_local(file_path):
    """Load data from a local file."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The dataset is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")
    
def preprocess_data(df):
    """Preprocess the data (e.g., handle missing values and rename columns)."""
    # Handle missing data (exclude columns with > 50% missing data)
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    
    # Check if the PatientID column exists, if not, create it
    if 'PatientID' not in df.columns:
        df['PatientID'] = range(len(df))
    
    # Retrieve the actual cryptic column names from the dataset
    cryptic_column_names = df.columns.tolist()
    
    # Desired readable column names #Hard coded

    readable_column_names = ['LCORE', 'LSURF', 'LO2', 'LBP', 'SURF_STBL', 'CORE_Stbl', 'BPS_STBL', 'COMFOR', 'DEC', 'PatientID'][:len(cryptic_column_names)]
    
    # Ensure the lengths match
    if len(cryptic_column_names) != len(readable_column_names):
        raise ValueError("The number of cryptic and readable column names does not match.")
    
    # Create the column mapping
    column_mapping = dict(zip(cryptic_column_names, readable_column_names))
    
    # Renaming columns according to the mapping
    df.rename(columns=column_mapping, inplace=True)
    print("Columns after renaming:", df.columns.tolist())  # Debugging statement to verify column renaming
    
    return df, df.columns.tolist()

# Step 2: Generate and Save Synthetic Data

def generate_synthetic_data(df, method='sample', n_samples=1000):
    """Generate synthetic data using the specified method."""
    if method == 'sample':
        synthetic_data = df.sample(n=n_samples, replace=True)
    elif method == 'ctgan':
        ctgan = CTGAN(epochs=10)  # You can adjust the number of epochs as needed
        discrete_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        ctgan.fit(df, discrete_columns)
        synthetic_data = ctgan.sample(n_samples)
    synthetic_data = synthetic_data.reset_index(drop=True)
    return synthetic_data

def save_synthetic_data(df, prefix='synthetic_data'):
    """Save synthetic data to a CSV file with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    return filename

def list_synthetic_files(directory='.'):
    """List saved synthetic data files in the directory."""
    files = [f for f in os.listdir(directory) if f.startswith('synthetic_data') and f.endswith('.csv')]
    return files

def clear_synthetic_files(directory='.'):
    """Clear saved synthetic data files in the directory."""
    files = list_synthetic_files(directory)
    for f in files:
        os.remove(os.path.join(directory, f))

# Function to find an available port

def find_available_port(start_port=8050, max_tries=100):
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1
    raise RuntimeError("No available ports found")

# Step 3: Load data and preprocess globally
#Other datasets at: https://data.world/search?context=community&q=cardiac+surgery+data+&type=resources
#
data_url = 'https://query.data.world/s/u33lunmprotq2ubl7nhhvg52txbjxg?dws=00000'  # Update this URL to the actual dataset URL
try:
    real_data, columns = preprocess_data(load_data(data_url))
    patient_ids = real_data['PatientID'].unique() if 'PatientID' in real_data.columns else []
except ValueError as e:
    print(e)
    columns = []
    patient_ids = []

# Step 4: Dashboard Design and Implementation

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For serving the file download

# Layout of the dashboard
app.layout = html.Div([
    html.Div([
        html.Label('Select X-axis variable:'),
        dcc.Dropdown(
            id='x-axis-selector',
            options=[{'label': var, 'value': var} for var in columns],
            value=columns[0] if columns else None  # Select the first variable by default
        ),
    ], style={'width': '32%', 'display': 'inline-block'}),
    html.Div([
        html.Label('Select Y-axis variable:'),
        dcc.Dropdown(
            id='y-axis-selector',
            options=[{'label': var, 'value': var} for var in columns],
            value=None  # No default value
        ),
    ], style={'width': '32%', 'display': 'inline-block'}),
    html.Div([
        html.Label('Select Plot Type:'),
        dcc.Dropdown(
            id='plot-type-selector',
            options=[
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Pie Chart', 'value': 'pie'}
            ],
            value='histogram'  # Default plot type
        ),
    ], style={'width': '32%', 'display': 'inline-block'}),
    html.Div([
        html.Label('Select Synthetic Data Generation Method:'),
        dcc.Dropdown(
            id='generation-method-selector',
            options=[
                {'label': 'Sample with Replacement', 'value': 'sample'},
                {'label': 'CTGAN', 'value': 'ctgan'}
            ],
            value='sample'  # Default method
        ),
    ], style={'width': '32%', 'display': 'inline-block'}),
    html.Button('Generate Synthetic Data', id='generate-synthetic-button', n_clicks=0),
    html.Button('Clear Synthetic Data', id='clear-synthetic-button', n_clicks=0),
    dcc.Dropdown(
        id='synthetic-selector',
        options=[{'label': f, 'value': f} for f in list_synthetic_files()],
        placeholder="Select synthetic dataset"
    ),
    html.Button('Download Selected Synthetic Data', id='download-synthetic-button', n_clicks=0),
    dcc.Download(id='download-synthetic-data'),
        html.Div([
        html.Label('Upload Local Data:'),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False  # Allow only a single file to be uploaded
        ),
    ], style={'width': '32%', 'display': 'inline-block'}),
    html.Div([
        html.Div([
            html.H3('Real Data'),
            dcc.Graph(id='real-variable-distribution'),
            dash_table.DataTable(id='real-summary-statistics')
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H3('Synthetic Data'),
            dcc.Graph(id='synthetic-variable-distribution'),
            dash_table.DataTable(id='synthetic-summary-statistics')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    html.Div([
        html.H3('Statistical Comparison'),
        dcc.Dropdown(
            id='stat-test-selector',
            options=[
                {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'},
                {'label': 'T-test', 'value': 'ttest'},
                {'label': 'Chi-Square Test', 'value': 'chi2'}
            ],
            placeholder="Select statistical test"
        ),
        html.Div(id='statistical-test-results')
    ]),
    html.Div([
        html.H3('Compare Individual Patient with Group'),
        dcc.Dropdown(
            id='patient-selector',
            options=[{'label': patient_id, 'value': patient_id} for patient_id in patient_ids],
            placeholder="Select patient"
        ),
        dcc.Dropdown(
            id='variable-selector',
            options=[{'label': var, 'value': var} for var in columns if var != 'PatientID'],
            placeholder="Select variable"
        ),
        dcc.Graph(id='individual-comparison')
    ])
])


# Combined callback to generate, save, and clear synthetic data 
@app.callback(
    Output('synthetic-selector', 'options'),
    [Input('generate-synthetic-button', 'n_clicks'),
     Input('clear-synthetic-button', 'n_clicks')],
    [State('x-axis-selector', 'value'),
     State('generation-method-selector', 'value')]
)
def manage_synthetic_data(generate_n_clicks, clear_n_clicks, generation_method):
    ctx = dash.callback_context

    if not ctx.triggered:
        return [{'label': f, 'value': f} for f in list_synthetic_files()]

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'generate-synthetic-button' and generate_n_clicks > 0:
        df_real, _ = preprocess_data(load_data(data_url))
        df_synthetic = generate_synthetic_data(df_real, method=generation_method)
        filename = save_synthetic_data(df_synthetic)
    elif button_id == 'clear-synthetic-button' and clear_n_clicks > 0:
        clear_synthetic_files()

    return [{'label': f, 'value': f} for f in list_synthetic_files()]

# Callback to serve the selected synthetic dataset file for download
@app.callback(
    Output('download-synthetic-data', 'data'),
    Input('download-synthetic-button', 'n_clicks'),
    State('synthetic-selector', 'value')
)
def download_synthetic_data(n_clicks, synthetic_file):
    if n_clicks > 0 and synthetic_file:
        path = os.path.join('.', synthetic_file)
        if os.path.exists(path):
            return dcc.send_file(path)
    raise PreventUpdate

# Callback to Handle File Uploads to update graphs and summary statistics for real and synthetic data
@app.callback(
    [Output('real-variable-distribution', 'figure'),
     Output('real-summary-statistics', 'data'),
     Output('real-summary-statistics', 'columns'),
     Output('synthetic-variable-distribution', 'figure'),
     Output('synthetic-summary-statistics', 'data'),
     Output('synthetic-summary-statistics', 'columns'),
     Output('statistical-test-results', 'children')],
    [Input('x-axis-selector', 'value'),
     Input('y-axis-selector', 'value'),
     Input('plot-type-selector', 'value'),
     Input('synthetic-selector', 'value'),
     Input('stat-test-selector', 'value'),
     Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)

def update_graphs(x_var, y_var, plot_type, synthetic_file, stat_test, contents, filename):
    if not x_var:
        return {}, [], [], {}, [], [], "Please select an x-axis variable."

    # Load real data
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_real = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
        df_real, _ = preprocess_data(load_data(data_url))

    # Load selected synthetic data
    if synthetic_file:
        df_synthetic = pd.read_csv(synthetic_file)
    else:
        df_synthetic = generate_synthetic_data(df_real)
    
    # Generate summary statistics
    selected_vars = [x_var]
    if y_var:
        selected_vars.append(y_var)
        
    real_summary_stats = df_real[selected_vars].describe(include='all')
    synthetic_summary_stats = df_synthetic[selected_vars].describe(include='all')

    real_summary_stats_data = real_summary_stats.reset_index().to_dict('records')
    real_summary_stats_columns = [{"name": i, "id": i} for i in real_summary_stats.reset_index().columns]
    
    synthetic_summary_stats_data = synthetic_summary_stats.reset_index().to_dict('records')
    synthetic_summary_stats_columns = [{"name": i, "id": i} for i in synthetic_summary_stats.reset_index().columns]

    # Extract categories from real data for categorical variables
    category_orders = {var: df_real[var].astype('category').cat.categories.tolist() for var in selected_vars if df_real[var].dtype.name == 'category' or df_real[var].dtype.name == 'object'}

    # Create histograms or pie charts for real data
    if plot_type == 'histogram':
        if y_var:
            fig_real = px.histogram(df_real, x=x_var, color=y_var, title=f'{x_var} vs {y_var} Distribution (Real)', category_orders=category_orders)
            fig_synthetic = px.histogram(df_synthetic, x=x_var, color=y_var, title=f'{x_var} vs {y_var} Distribution (Synthetic)', category_orders=category_orders)
        else:
            fig_real = px.histogram(df_real, x=x_var, title=f'{x_var} Distribution (Real)', category_orders=category_orders)
            fig_synthetic = px.histogram(df_synthetic, x=x_var, title=f'{x_var} Distribution (Synthetic)', category_orders=category_orders)
    elif plot_type == 'pie':
        fig_real = px.pie(df_real, names=x_var, title=f'{x_var} Distribution (Real)')
        fig_synthetic = px.pie(df_synthetic, names=x_var, title=f'{x_var} Distribution (Synthetic)')

    # Perform statistical test
    if stat_test and len(selected_vars) == 1:
        if df_real[x_var].dtype.kind in 'iufc' and df_synthetic[x_var].dtype.kind in 'iufc':
            real_data = df_real[x_var].dropna()
            synthetic_data = df_synthetic[x_var].dropna()
            if stat_test == 'ks':
                stat, p_value = ks_2samp(real_data, synthetic_data)
                test_result = f'Kolmogorov-Smirnov Test:\nStatistic: {stat}\nP-value: {p_value}'
            elif stat_test == 'ttest':
                stat, p_value = ttest_ind(real_data, synthetic_data)
                test_result = f'T-test:\nStatistic: {stat}\nP-value: {p_value}'
        elif df_real[x_var].dtype.name == 'category' or df_real[x_var].dtype.name == 'object':
            real_freq = df_real[x_var].value_counts()
            synthetic_freq = df_synthetic[x_var].value_counts()
            observed = np.array([real_freq, synthetic_freq])
            stat, p_value, dof, expected = chi2_contingency(observed, correction=False)
            test_result = f'Chi-Square Test:\nStatistic: {stat}\nP-value: {p_value}'
        else:
            test_result = "Unsupported variable type for statistical test."
    else:
        test_result = "Please select a statistical test and a single variable for comparison."

    return fig_real, real_summary_stats_data, real_summary_stats_columns, fig_synthetic, synthetic_summary_stats_data, synthetic_summary_stats_columns, test_result

# Callback to update individual patient comparison
@app.callback(
    Output('individual-comparison', 'figure'),
    [Input('patient-selector', 'value'),
     Input('variable-selector', 'value')]
)
def update_individual_comparison(patient_id, variable):
    if not patient_id or not variable:
        raise PreventUpdate

    # Load and preprocess real data
    df_real, _ = preprocess_data(load_data(data_url))

    # Get the patient's value
    patient_value = df_real[df_real['PatientID'] == patient_id][variable].values[0]

    # Create boxplot
    fig = px.box(df_real, y=variable, title=f'{variable} Comparison for Patient {patient_id}')
    fig.add_trace(go.Scatter(x=[f'PatientID = {patient_id}'], y=[patient_value], mode='markers', marker=dict(color='red', size=10), name=f'PatientID = {patient_id}'))

    return fig

# Step 4: Run the Dash Application

if __name__ == '__main__':
    port = find_available_port()
    app.run_server(debug=True, port=port)
