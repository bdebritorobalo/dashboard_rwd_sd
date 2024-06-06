# File path: dashboard_analysis.py

import base64
import dash
import datetime
import dash
import dash_bootstrap_components as dbc
import itertools
import numpy as np
import os
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import socket
from ctgan import CTGAN
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import send_file
from scipy.stats import ks_2samp, ttest_ind, chi2_contingency

# Step 1: Load and Preprocess Data

def load_data(contents, filename):
    """Load data from a local CSV file."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            raise ValueError("Unsupported file type.")
        if df.empty:
            raise ValueError("The dataset is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from {filename}: {e}")

def preprocess_data(df):
    """Preprocess the data (e.g., handle missing values)."""
    df = df.dropna(thresh=len(df) * 0.5, axis=1)
    return df, df.columns.tolist()

# Step 2: Generate and Save Synthetic Data

def generate_synthetic_data(df, method='sample', n_samples=500):
    """Generate synthetic data using the specified method."""
    if method == 'sample':
        synthetic_data = df.sample(n=n_samples, replace=True)
    elif method == 'ctgan':
        ctgan = CTGAN(epochs=300)
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


# Boxplot data manipulation
def create_plot_data(data, procedures, column):        #! USING CALLBACKS
    '''Data needs to be inserted in specific format to make grouped boxes.
    !This version will only work if the y-axis is a time measurement in minutes!
    :param: data: dataframe with all pre-processed data
    :param: procedures: From callback, the procedures that will be listed on x-axis
    :param: column: Column name (for durations) that should be added for each procedure.'''
    y_data=[]
    x_names=[]

    df_dict = pd.DataFrame({'name':['Norwood', 'Glenn', 'Adapted Fontan'], 'code':['333024B', '333226','333025']})

    for proc in procedures:
        data_temp = data[column].loc[data['procedure_code'] == proc]
        y_data.extend(data_temp)
        x_names.extend(list(itertools.repeat(df_dict['name'].loc[df_dict['code'] == proc].item(),
                                             len(data_temp))))
    return y_data, x_names




# Step 3: Dashboard Design and Implementation

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # For serving the file download

# --------------- Layout of the dashboard -------------------- #
app.layout = html.Div([
    #HEADER
    html.Div([
                        html.Img(src='/assets/phems_logo_RGB_color_cropped.png'),
                        html.Div([
                            html.H2('Dashboard: Real world data vs. Synthetic data', className='header-title')
                            ],className='header-title-container')
                        ],className= 'header'),
    html.Div(style={'height': '110px'}),    #spacer for content and header
    html.Div([html.H4('Upload both files first:', style={'text-align':'center', 'margin':'10px'})]),
    # fix follows later

    html.Div([
        html.Div([
            html.Label('Upload CSV File - Medical data:', style={'margin-left': '2.5%'}),
            dcc.Upload(
                id='upload-data-medical',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files'),
                ], className='upload-box'),
                multiple=False
            ),
            html.Div(id='output-data-upload-medical', className='table_container')
        ],className='upload-box-container'),

    html.Div([
        html.Label('Upload CSV File - Synthetic data:', style={'margin-left': '2.5%'}),
        dcc.Upload(
            id='upload-data-synth',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files'),
            ], className='upload-box'),
            multiple=False
        ),
        html.Div(id='output-data-upload-synth', className='table_container')
    ],className='upload-box-container')
    ]),

    # Bruno's settings
    html.Div([
        html.Label('Select X-axis variable:'),
        dcc.Dropdown(
            id='x-axis-selector',
            options=[],
            value=None
        ),
    ], style={'width': '32%', 'display': 'inline-block', 'margin-left': '1%'}),
    html.Div([
        html.Label('Select Y-axis variable:'),
        dcc.Dropdown(
            id='y-axis-selector',
            options=[],
            value=None
        ),
    ], style={'width': '32%', 'display': 'inline-block', 'margin-left': '1%'}),
    html.Div([
        html.Label('Select Plot Type:'),
        dcc.Dropdown(
            id='plot-type-selector',
            options=[
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Pie Chart', 'value': 'pie'},
                {'label': 'Box Plot', 'value': 'box'},
                {'label': 'Scatter Plot', 'value': 'scatter'}
            ],
            value='histogram'  # Default plot type
        ),
    ], style={'width': '32%', 'display': 'inline-block', 'margin-left': '1%'}),

    # html.Div([
    #     html.Label('Select Synthetic Data Generation Method:'),
    #     dcc.Dropdown(
    #         id='generation-method-selector',
    #         options=[
    #             {'label': 'Sample with Replacement', 'value': 'sample'},
    #             {'label': 'CTGAN', 'value': 'ctgan'}
    #         ],
    #         value='sample'  # Default method
    #     ),
    # ], style={'width': '32%', 'display': 'inline-block'}),

    # html.Div([
    #     html.Label('Select Color Scheme:'),
    #     dcc.Dropdown(
    #         id='color-scheme-selector',
    #         options=[

    #             {'label': 'Plotly', 'value': 'Plotly'}
    #         ],
    #         value='Plotly'  # Default color scheme
    #     ),
    # ], style={'width': '32%', 'display': 'inline-block'}),

    # html.Button('Generate Synthetic Data', id='generate-synthetic-button', n_clicks=0),
    # html.Button('Clear Synthetic Data', id='clear-synthetic-button', n_clicks=0),
    # dcc.Dropdown(
    #     id='synthetic-selector',
    #     options=[{'label': f, 'value': f} for f in list_synthetic_files()],
    #     placeholder="Select synthetic dataset"
    # ),
    # html.Button('Download Selected Synthetic Data', id='download-synthetic-button', n_clicks=0),
    # dcc.Download(id='download-synthetic-data'),
    html.Div([
        html.Div([
            html.H3('Real Data'),
            dcc.Graph(id='real-variable-distribution'),
            dash_table.DataTable(id='real-summary-statistics')
        ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '20px'}),
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
    html.Div([],style={'height':'50px'}),

    html.Div([
    html.Div([html.H3(['Boxplot'])], style={'margin-left':'1%'}),
    html.Div([
        html.Label('Select procedure(s):'),
        dcc.Dropdown(
            id='procedure-selector',
            options=[],
            value=None,
            multi=True
            )
        ], style={'width': '48%', 'display': 'inline-block', 'margin-left':'1%'}),   #, 'margin-left':'1%'
    html.Div([
        html.Label('Select column(s):'),
        dcc.Dropdown(
            id='plot-type-selector',
            options=[
                {'label': 'Histogram', 'value': 'histogram'},
                {'label': 'Pie Chart', 'value': 'pie'},
                {'label': 'Box Plot', 'value': 'box'},
                {'label': 'Scatter Plot', 'value': 'scatter'}
            ],
            value='ECC_duration',  # Default plot type
            multi=True
            ),
        ], style={'width': '48%', 'display':'inline-block', 'margin-left':'1%'}),      #, 'margin-left':'1%'

        dcc.Graph(id='patient-comparison-boxplot')
    ]),


    html.Div([
        html.H3('Patient Comparison'),
        html.Label('Select Patient:'),
        dcc.Dropdown(
            id='patient-selector',
            options=[],
            value=None
        ),
        dcc.Graph(id='patient-comparison-boxplot')
    ])
])

# Update column selectors based on uploaded data
@app.callback(
    [Output('x-axis-selector', 'options'),
     Output('y-axis-selector', 'options'),
     Output('patient-selector', 'options'),
     Output('output-data-upload-medical', 'children'),
     Output('output-data-upload-synth', 'children')],
    [Input('upload-data-medical', 'contents'),
     Input('upload-data-synth', 'contents')],
    [State('upload-data-medical', 'filename'),
     State('upload-data-synth', 'filename')]
)
def update_columns(contents_medical, contents_synth, filename_medical, filename_synth):
    '''says that its missing docstring, anoying'''

    if contents_medical is None:
        raise PreventUpdate

    #load and preprocess real-world data
    df_medical = load_data(contents_medical, filename_medical)
    _, columns_medical = preprocess_data(df_medical)

    options = [{'label': col, 'value': col} for col in columns_medical]
    patient_options = [{'label': f'Patient {i}', 'value': i} for i in df_medical.index]
    table_medical = dash_table.DataTable(
        data=df_medical.head().to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_medical.columns],
        page_size=10,
        style_table={'overflowX': 'scroll', 'width':'45%', 'margin':'0 1%'}
        )

    # if contents_synth is None:
    #     raise PreventUpdate

    # Add synthetic data with the same columns
    df_synth = load_data(contents_synth, filename_synth)
    table_synth = dash_table.DataTable(
        data=df_synth.head().to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_synth.columns],
        page_size=10,
        style_table={'overflowX': 'scroll', 'width':'45%', 'margin':'0 1%'}
        )
    return options, options, patient_options, table_medical, table_synth


# Combined callback to generate, save, and clear synthetic data
# @app.callback(
#     Output('synthetic-selector', 'options'),
#     [Input('generate-synthetic-button', 'n_clicks'),
#      Input('clear-synthetic-button', 'n_clicks')],
#     [State('generation-method-selector', 'value'),
#      State('upload-data', 'contents'),
#      State('upload-data', 'filename')]
# )
# def manage_synthetic_data(generate_n_clicks, clear_n_clicks, generation_method, contents, filename):
#     ctx = dash.callback_context

#     if not ctx.triggered:
#         return [{'label': f, 'value': f} for f in list_synthetic_files()]

#     button_id = ctx.triggered[0]['prop_id'].split('.')[0]

#     if button_id == 'generate-synthetic-button' and generate_n_clicks > 0:
#         df_real = load_data(contents, filename)
#         df_real, _ = preprocess_data(df_real)
#         df_synthetic = generate_synthetic_data(df_real, method=generation_method)
#         save_synthetic_data(df_synthetic)
#     elif button_id == 'clear-synthetic-button' and clear_n_clicks > 0:
#         clear_synthetic_files()

#     return [{'label': f, 'value': f} for f in list_synthetic_files()]

# # Callback to serve the selected synthetic dataset file for download
# @app.callback(
#     Output('download-synthetic-data', 'data'),
#     Input('download-synthetic-button', 'n_clicks'),
#     State('synthetic-selector', 'value')
# )
# def download_synthetic_data(n_clicks, synthetic_file):
#     if n_clicks > 0 and synthetic_file:
#         path = os.path.join('.', synthetic_file)
#         if os.path.exists(path):
#             return send_file(path)
#     raise PreventUpdate

# Callback to update graphs and summary statistics for real and synthetic data
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
    #  Input('color-scheme-selector', 'value'),
     Input('stat-test-selector', 'value'),
     Input('upload-data-medical', 'contents'),
     Input('upload-data-medical', 'filename'),
     Input('upload-data-synth', 'contents'),
     Input('upload-data-synth', 'filename')]
)
def update_graphs(x_var, y_var, plot_type, stat_test, contents_medical, filename_medical, contents_synth, filename_synth):
    if not x_var:
        return {}, [], [], {}, [], [], "Please select an x-axis variable."

    # Load and preprocess real data
    df_real = load_data(contents_medical, filename_medical)
    df_real, _ = preprocess_data(df_real)

    # Load selected synthetic data
    if contents_synth and filename_synth:
        df_synthetic = load_data(contents_synth, filename_synth)
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

    # Set color scheme
    #TODO: MAKE PHEMS COLORSCHEME --> Colors are already available in BOXPLOT JUPYTER
    # if color_scheme == 'default':
    #     color_sequence = px.colors.sequential.Viridis
    # else:
    #     try:
    #         color_sequence = getattr(px.colors.sequential, color_scheme)
    #     except AttributeError:
    #         color_sequence = getattr(px.colors.qualitative, color_scheme)

    # Create the plot
    fig_real, fig_synthetic = None, None

    if plot_type == 'histogram':
        if y_var:
            fig_real = px.histogram(df_real, x=x_var, color=y_var, title=f'{x_var} vs {y_var} Distribution (Real)', category_orders=category_orders) #, color_discrete_sequence=color_sequence)
            fig_synthetic = px.histogram(df_synthetic, x=x_var, color=y_var, title=f'{x_var} vs {y_var} Distribution (Synthetic)', category_orders=category_orders) #, color_discrete_sequence=color_sequence)
        else:
            fig_real = px.histogram(df_real, x=x_var, title=f'{x_var} Distribution (Real)', category_orders=category_orders) #, color_discrete_sequence=color_sequence)
            fig_synthetic = px.histogram(df_synthetic, x=x_var, title=f'{x_var} Distribution (Synthetic)', category_orders=category_orders) #, color_discrete_sequence=color_sequence)
    elif plot_type == 'pie':
        fig_real = px.pie(df_real, names=x_var, title=f'{x_var} Distribution (Real)')#, color_discrete_sequence=color_sequence)
        fig_synthetic = px.pie(df_synthetic, names=x_var, title=f'{x_var} Distribution (Synthetic)')#, color_discrete_sequence=color_sequence)
    elif plot_type == 'box':
        if y_var:
            fig_real = px.box(df_real, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Distribution (Real)')#, color_discrete_sequence=color_sequence)
            fig_synthetic = px.box(df_synthetic, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Distribution (Synthetic)')#, color_discrete_sequence=color_sequence)
        else:
            fig_real = px.box(df_real, x=x_var, title=f'{x_var} Distribution (Real)')#, color_discrete_sequence=color_sequence)
            fig_synthetic = px.box(df_synthetic, x=x_var, title=f'{x_var} Distribution (Synthetic)')#, color_discrete_sequence=color_sequence)
    elif plot_type == 'scatter':
        if y_var:
            fig_real = px.scatter(df_real, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Scatter Plot (Real)')#, color_discrete_sequence=color_sequence)
            fig_synthetic = px.scatter(df_synthetic, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Scatter Plot (Synthetic)')#, color_discrete_sequence=color_sequence)
        else:
            fig_real = px.scatter(df_real, x=x_var, title=f'{x_var} Scatter Plot (Real)')#, color_discrete_sequence=color_sequence)
            fig_synthetic = px.scatter(df_synthetic, x=x_var, title=f'{x_var} Scatter Plot (Synthetic)')#, color_discrete_sequence=color_sequence)

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

# # Callback to update the patient comparison boxplot
# @app.callback(
#     Output('patient-comparison-boxplot', 'figure'),
#     [Input('patient-selector', 'value'),
#      Input('x-axis-selector', 'value'),
#      Input('synthetic-selector', 'value'),
#      Input('upload-data', 'contents'),
#      Input('upload-data', 'filename')]
# )
# def update_patient_comparison_boxplot(patient_idx, x_var, synthetic_file, contents, filename):
#     if not x_var or patient_idx is None:
#         raise PreventUpdate

#     # Load and preprocess real data
#     df_real = load_data(contents, filename)
#     df_real, _ = preprocess_data(df_real)

#     # Load selected synthetic data
#     if synthetic_file:
#         df_synthetic = pd.read_csv(synthetic_file)
#     else:
#         df_synthetic = generate_synthetic_data(df_real)

#     patient_value = df_real.loc[patient_idx, x_var]

#     fig = go.Figure()

#     fig.add_trace(go.Box(
#         y=df_real[x_var],
#         name="Real Data",
#         boxpoints='outliers'
#     ))

#     fig.add_trace(go.Box(
#         y=df_synthetic[x_var],
#         name="Synthetic Data",
#         boxpoints='outliers'
#     ))

#     fig.add_trace(go.Scatter(
#         x=['Real Data', 'Synthetic Data'],
#         y=[patient_value, patient_value],
#         mode='markers+text',
#         name='Selected Patient',
#         text=['Patient'] * 2,
#         textposition='top center'
#     ))

#     fig.update_layout(
#         title=f'Patient {patient_idx} Comparison on {x_var}',
#         yaxis_title=x_var
#     )

#     return fig

# Step 4: Run the Dash Application

if __name__ == '__main__':
    port = find_available_port()
    app.run_server(debug=True, port=port)
