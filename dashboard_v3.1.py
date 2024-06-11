
'''
Python: dashboard to compare real-world data to synthetic data using Dash Plotly.
Use this file as a main.py
'''

import base64
import itertools
import io
import socket
import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from ctgan import CTGAN
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# from flask import send_file
from scipy.stats import ks_2samp, ttest_ind, chi2_contingency

# Step 1: Load and Preprocess Data
# phems_colors = ['#4e8ca9', '#D18E46', '#5BB6AB', '#A6CFDF']

def load_data(contents, filename):
    """Load data from a local CSV file."""
    _, content_string = contents.split(',')     # _ used to be content_type
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


def create_plot_data(data, procedures, column, main_proc):
    '''Create data in form of lists, including all procedures to be able to plot grouped box-plots'''
    y_data=[]
    x_names=[]

    # needs to be expanded --> wait for synthetic data how this could be done optimally
    # df_dict = pd.DataFrame({'name':['Norwood', 'Glenn', 'Adapted Fontan','ECMO'],
    #   'code':['333024B', '333226','333025', '333180']})
    df_dict = data[['procedure_code', 'procedure_text']].drop_duplicates()

    if main_proc:
        # print('main_proc is aan')
        data = data.loc[data['main_procedure'] == 1]
    # print(len(data))

    for proc in procedures:
        data_temp = data[column].loc[data['procedure_code'] == proc]
        y_data.extend(data_temp)
        x_names.extend(list(itertools.repeat(df_dict['procedure_text'].loc[df_dict['procedure_code'] == proc].item(),
                                             len(data_temp))))
    return y_data, x_names

# Function to find an available port

def find_available_port(start_port=8050, max_tries=100):
    """Find an available port starting from start_port."""
    port_attempt = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port_attempt)) != 0:
                return port_attempt
            port_attempt += 1
    raise RuntimeError("No available ports found")




# Step 3: Dashboard Design and Implementation

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # For serving the file download

# --------------- Layout of the dashboard -------------------- #
app.layout = html.Div([
    #HEADER
    html.Div([
        html.Img(src='/assets/phems_logo_RGB_color_cropped.png'),
        html.H2('Dashboard: Real world data vs. Synthetic data', className='header-title'),
        ],className= 'header'),
    html.Div(style={'height': '110px'}),                                #spacer for content and header
    html.Div([html.H4('Upload both files before continuing:', style={'text-align':'center'})]),
    html.Div([
        html.Div([
            html.Label('Upload CSV File - Medical data:'),
            dcc.Upload(
                id='upload-data-medical',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files'),
                ], className='upload-box'),
                multiple=False
            ),
            html.Div(id='output-data-upload-medical', className='table-container')
        ], className='upload-internal-container'),

    html.Div([
        html.Label('Upload CSV File - Synthetic data:'),
        dcc.Upload(
            id='upload-data-synth',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files'),
            ], className='upload-box'),
            multiple=False
        ),
        html.Div(id='output-data-upload-synth', className='table-container')
    ], className='upload-internal-container')
    ], className='upload-box-container'),

    # Bruno's settings
   html.Div([html.Div([
        html.Label('Select X-axis variable:'),
        dcc.Dropdown(
            id='x-axis-selector',
            options=[],
            value=None
        ),
    ], className='drop-box'),
    html.Div([
        html.Label('Select Y-axis variable:'),
        dcc.Dropdown(
            id='y-axis-selector',
            options=[],
            value=None
        ),
    ], className='drop-box'),
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
    ], className='drop-box'),], className='drop-box-container'),

# comparing 2 graphs!
    html.Div([
        html.Div([
            html.H4('Real Data'),
            dcc.Graph(id='real-variable-distribution'),
            dash_table.DataTable(id='real-summary-statistics')
        ], className='compare-box'),
        html.Div([
            html.H4('Synthetic Data'),
            dcc.Graph(id='synthetic-variable-distribution'),
            dash_table.DataTable(id='synthetic-summary-statistics')
        ], className='compare-box' ) #style={'width': '48%', 'display': 'inline-block','outline':'solid lime'})
    ], className='compare-container'),

    html.Div([
        html.H3('Statistical Comparison'),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='stat-test-selector',
                    options=[
                        {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'},
                        {'label': 'T-test', 'value': 'ttest'},
                        {'label': 'Chi-Square Test', 'value': 'chi2'}
                    ], placeholder="Select statistical test"
                    ),
            ], className= 'drop-box'),
            html.Div(id='statistical-test-results', className = 'statistics')
        ], className='drop-box-container'),
    ]),
    html.Div([
        html.Div([html.H3(['Boxplot'])]),
        html.Div([
            html.Div([
                html.Label('Select procedure(s):'),
                dcc.Dropdown(
                    id='procedure-selector',
                    options=[],
                    value= None,
                    multi=True
                    )
                ], className='dropdown-menu-container'),
            html.Div([
                html.Label('Select column(s):'),
                dcc.Dropdown(
                    id='column-selector',
                    options=[
                        {'label': 'Procedure Duration', 'value': 'procedure_duration'},
                        {'label': 'Extra Corporal Circulation', 'value': 'ECC_duration'},
                        {'label': 'Aorta Clamping Time', 'value': 'AOX_duration'},
                        {'label': 'Deep Hypothermic Cardiac Arrest', 'value': 'DHCA_duration'},
                    ],
                    value= None,  # Default plot type
                    multi=True
                    ),
                ], className='dropdown-menu-container'),
            html.Div([
                html.Label('Toggle for main-procedures only:'),
                daq.BooleanSwitch(id='main-proc-toggle',
                    on=False,
                    color='#4e8ca9',
                    className= 'boolean-button'
                    ),
                ], className='button-menu-container')
        ], className='box-menu-container')
    ]),

    html.Div([
        html.Div([
            html.H4('Real Data'),
            dcc.Graph(id='medical-patient-boxplot'),
        ], className='compare-box'),
        html.Div([
            html.H4('Synthetic Data'),
            dcc.Graph(id='synthetic-boxplot'),
        ], className='compare-box' )
    ], className='compare-container'),
]),


# Update column selectors based on uploaded data
@app.callback(
    [Output('x-axis-selector', 'options'),
     Output('y-axis-selector', 'options'),
     Output('procedure-selector', 'options'),
    #  Output('patient-selector', 'options'),
     Output('output-data-upload-medical', 'children'),
     Output('output-data-upload-synth', 'children')],
    [Input('upload-data-medical', 'contents'),
     Input('upload-data-synth', 'contents')],
    [State('upload-data-medical', 'filename'),
     State('upload-data-synth', 'filename')]
)
def update_columns(contents_medical, contents_synth, filename_medical, filename_synth):
    '''says that its missing docstring, annoying'''

    if contents_medical is None:
        raise PreventUpdate

    #load and preprocess real-world data
    df_medical = load_data(contents_medical, filename_medical)
    _, columns_medical = preprocess_data(df_medical)

    options = [{'label': col, 'value': col} for col in columns_medical]
    procedures = [{'label': procedure_code, 'value': procedure_code}
                  for  procedure_code in df_medical['procedure_code'].unique()]

    # patient_options = [{'label'f: f'Patient {i}', 'value': i} for i in df_medical.index]
    table_medical = dash_table.DataTable(
        data=df_medical.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_medical.columns],
        page_size= 8,
        filter_action='native',
        filter_options= {'case':'insensitive'},
        style_table={'overflowX': 'auto', 'virtualization':'True'},
        style_header={'fontWeight':'900'},
        style_cell={'fontWeight':'100'},
        )

    if contents_synth is None:
        raise PreventUpdate

    # Add synthetic data with the same columns
    df_synth = load_data(contents_synth, filename_synth)
    table_synth = dash_table.DataTable(
        data=df_synth.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_synth.columns],
        page_size= 8,
        filter_action= 'native',
        filter_options= {'case':'insensitive'},
        style_table={'overflowX': 'auto', 'virtualization':'True'},
        style_header={'font-weight':'900'},
        style_cell={'font-weight':'100'},
        )
    return options, options, procedures, table_medical, table_synth
    # return options, options, procedures, patient_options, table_medical, table_synth


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
def update_graphs(x_var, y_var, plot_type, stat_test, contents_medical,
                  filename_medical, contents_synth, filename_synth):
    '''Create the first section of graphs: histo,pie,box,scatter with statistics'''
    if not x_var:
        return {}, [], [], {}, [], [], "Please select an x-axis variable."

    # Load and preprocess real data
    df_real = load_data(contents_medical, filename_medical)
    # Load selected synthetic data
    df_synthetic = load_data(contents_synth, filename_synth)

    # Generate summary statistics
    selected_vars = [x_var]
    if y_var:
        selected_vars.append(y_var)

    real_summary_stats = df_real[selected_vars].describe(include='all')
    synthetic_summary_stats = df_synthetic[selected_vars].describe(include='all')

    real_summary_stats_data = real_summary_stats.reset_index().to_dict('records')
    real_summary_stats_columns = [{"name": i, "id": i}
                                  for i in real_summary_stats.reset_index().columns]

    synthetic_summary_stats_data = synthetic_summary_stats.reset_index().to_dict('records')
    synthetic_summary_stats_columns = [{"name": i, "id": i}
                                       for i in synthetic_summary_stats.reset_index().columns]

    # Extract categories from real data for categorical variables
    category_orders = {var: df_real[var].astype('category').cat.categories.tolist()
                       for var in selected_vars if df_real[var].dtype.name == 'category' or df_real[var].dtype.name == 'object'}

# Create the plot
    fig_real, fig_synthetic = None, None
    phems_colors = ['#4e8ca9', '#D18E46', '#5BB6AB', '#A6CFDF'] # used to be:color_sequence) 


    if plot_type == 'histogram':
        if y_var:
            fig_real = px.histogram(df_real, x=x_var, color=y_var, title=f'{x_var} vs {y_var} Distribution (Real)', category_orders=category_orders, color_discrete_sequence=phems_colors)
            fig_synthetic = px.histogram(df_synthetic, x=x_var, color=y_var, title=f'{x_var} vs {y_var} Distribution (Synthetic)', category_orders=category_orders , color_discrete_sequence=phems_colors)
        else:
            fig_real = px.histogram(df_real, x=x_var, title=f'{x_var} Distribution (Real)', category_orders=category_orders, color_discrete_sequence=phems_colors)
            fig_synthetic = px.histogram(df_synthetic, x=x_var, title=f'{x_var} Distribution (Synthetic)', category_orders=category_orders, color_discrete_sequence=phems_colors)
    elif plot_type == 'pie':
        fig_real = px.pie(df_real, names=x_var, title=f'{x_var} Distribution (Real)', color_discrete_sequence=phems_colors)
        fig_synthetic = px.pie(df_synthetic, names=x_var, title=f'{x_var} Distribution (Synthetic)', color_discrete_sequence=phems_colors)
    elif plot_type == 'box':
        if y_var:
            fig_real = px.box(df_real, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Distribution (Real)', color_discrete_sequence=phems_colors )
            fig_synthetic = px.box(df_synthetic, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Distribution (Synthetic)', color_discrete_sequence=phems_colors)
        else:
            fig_real = px.box(df_real, x=x_var, title=f'{x_var} Distribution (Real)', color_discrete_sequence=phems_colors)
            fig_synthetic = px.box(df_synthetic, x=x_var, title=f'{x_var} Distribution (Synthetic)', color_discrete_sequence=phems_colors)
    elif plot_type == 'scatter':
        if y_var:
            fig_real = px.scatter(df_real, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Scatter Plot (Real)', color_discrete_sequence=phems_colors)
            fig_synthetic = px.scatter(df_synthetic, x=x_var, y=y_var, title=f'{x_var} vs {y_var} Scatter Plot (Synthetic)', color_discrete_sequence=phems_colors)
        else:
            fig_real = px.scatter(df_real, x=x_var, title=f'{x_var} Scatter Plot (Real)', color_discrete_sequence=phems_colors)
            fig_synthetic = px.scatter(df_synthetic, x=x_var, title=f'{x_var} Scatter Plot (Synthetic)', color_discrete_sequence=phems_colors)

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
            stat, p_value, _, _ = chi2_contingency(observed, correction=False)     # _1 = dof, _2 = expected
            test_result = f'Chi-Square Test:\nStatistic: {stat}\nP-value: {p_value}'
        else:
            test_result = "Unsupported variable type for statistical test."
    else:
        test_result = "Please select a statistical test and a single variable for comparison."

    return fig_real, real_summary_stats_data, real_summary_stats_columns, fig_synthetic, synthetic_summary_stats_data, synthetic_summary_stats_columns, test_result

# # Callback to update the patient comparison boxplot
@app.callback(
    [Output('medical-patient-boxplot', 'figure'),
    Output('synthetic-boxplot', 'figure')],
    [Input('procedure-selector', 'value'),
     Input('column-selector', 'value'),
     Input('upload-data-medical','contents'),
     Input('upload-data-synth', 'contents'),
     Input('upload-data-medical', 'filename'),
     Input('upload-data-synth', 'filename'),
     Input('main-proc-toggle', 'on')]
)
def update_patient_comparison_boxplot(procedures, columns, contents, synth_contents,filename, synth_filename, on):
    '''Docstring problem exists again: Create the box-plots'''

    phems_colors = ['#4e8ca9', '#D18E46', '#5BB6AB', '#A6CFDF']

    if not contents or not synth_contents:
        raise PreventUpdate

    df_real = load_data(contents, filename)
    df_synthetic = load_data(synth_contents, synth_filename)

    if not procedures or not columns:
        raise PreventUpdate

    # Create boxplot 1:
    fig_medical = go.Figure()
    for i, column in enumerate(columns):
        y_data, x_names = create_plot_data(data=df_real, procedures=procedures, column=column, main_proc=on)
        fig_medical.add_trace(go.Box(
            boxpoints='all',
            y= y_data,
            x= [(name[:14] + '...' + name[-14:]) if len(name) > 32 else name for name in x_names],
            name= f'{column}',
            marker_color= phems_colors[i]
        ))
    fig_medical.update_layout(
        yaxis_title='Duration in minutes',
        xaxis_title='Procedure codes',
        boxmode='group'  ,                  # group together boxes of the different traces for each value of x
        height=600
    )

# Create boxplot 2:
    fig_synth = go.Figure()
    for i, column in enumerate(columns):
        y_data, x_names = create_plot_data(data=df_synthetic, procedures=procedures, column=column, main_proc=on)
        fig_synth.add_trace(go.Box(
            boxpoints='all',
            y= y_data,
            x= [(name[:14] + '...' + name[-14:]) if len(name) > 40 else name for name in x_names],          #Shorten the names for plot visibility
            name= f'{column}',
            marker_color= phems_colors[i],
        ))
    fig_synth.update_layout(
        yaxis_title='Duration in minutes',
        xaxis_title='Procedure codes',
        boxmode='group',                        # group together boxes of the different traces for each value of x
        height=600
    )
    return fig_medical, fig_synth


# Step 4: Run the Dash Application
if __name__ == '__main__':
    app.run_server(debug=True, port= find_available_port())
