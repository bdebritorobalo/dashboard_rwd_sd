import pandas as pd 
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import socket


df = pd.read_csv('20240528_filter.csv', usecols= ['identifier_value', 'subject_Patient_value', 'time_OR', 'postop_diagnosis_code', 'postop_diagnosis_text', 'procedure_code', 'procedure_text', 'procedure_duration', 'age_procedure', 'status_sternum', 'ECC_duur_min', 'AoX_duur_min', 'Circulatory_arrest_min', 'Antegrade_circulation_min'])


def find_available_port(start_port=8050, max_tries=100):
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1
    raise RuntimeError("No available ports found")




app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# style={
#     'width': 450,
#     'margin-right': 35,
#     'margin-top': 35,
#     'margin-bototm': 35},

app.layout = dbc.Container([html.Div(children=[
    html.B('Menu', style={'margin_bottom': 20}),
    dcc.Dropdown(
        id='cohort',
        options=[{'label': procedure, 'value': procedure} for procedure in df['procedure_text'].unique()],
        multi=True,
        placeholder="Select procedure",
        style={'margin-bottom': 20}
    )
    ,dcc.Dropdown( #style={'marge-top':20},
        options = [{'label': col, 'value': col} for col in df.columns]
        ,clearable=True
        ,multi=True
        ,id='graph'
        ,placeholder='Select x-axis Box-plot'
        , className='navbar')
        ]
        )
    ,html.Div(children=[
            dcc.Graph(id='boxplot'),
            dash_table.DataTable(
                id='datatable',
                data = df.to_dict('records'),
                columns = [{'name': col, 'id': col} for col in df.columns],
                page_size= 10,
                style_table={'overflowX': 'scroll'})
        ] ,className='dashboard-container'
        )],
            fluid=True
            #className='dashboard-container')
            ,style={'display': 'flex'})


# 'width': 1100,
# 'margin-top': 35,
# 'margin-right': 35,
# 'margin-bottom': 35

#filter section
# app.layout = html.Div([
#     dcc.Dropdown(
#     id='cohort',
#     options=[{'label': procedure, 'value': procedure} for procedure in df['procedure_text'].unique()],
#     multi=True,
#     placeholder="Select procedure"
#     )
#     ,dcc.Dropdown(
#                     options = [{'label': col, 'value': col} for col in df.columns]
#             	    # options=['time_OR', 'procedure_duration', 'procedure_duration_fix', 'age_procedure', 'status_sternum', 'age_procedure', 'ECC_duur_min', 'AoX_duur_min', 'Circulatory_arrest_min', 'Antegrade_circulation_min']
#                 #  ['perimembraneus ventrikelseptumdefect','HLHS', 'Tetralogy of Fallot', 'Transp. Great Arteries', 'Coarctatio aortae']
#                 #  ,label= [10567, 10596, 10579, 61727, 10606]
#                  ,clearable=True
#                  ,multi=False
#                  ,id='graph'
#                  ,placeholder='Select x-axis barchart'
#                 )

#     # ,html.Div(id='dd-output-container')
#     ,dash_table.DataTable(
#         id='datatable',
#         data = df.to_dict('records'), 
#         columns = [{'name': col, 'id': col} for col in df.columns], 
#         page_size= 10)
#     ,dcc.Graph(id='boxplot')
#     # ,html.Div(
#     #     dash_table.DataTable(
#     #     id='selected_data',
#     #     data = df.to_dict('records'), 
#     #     columns = [{'name': col, 'id': col} for col in df.columns], 
#     #     page_size= 10),


#     #     style={'marginTop': 20})

# ])

# @callback(Output('dd-output-container', 'children'), Input('cohort', 'value'), Iput )
# def subselection_df(value):
#     df = df[df['postop_diagnosis_code'] == value]
#     return value# , df

@app.callback(
    [Output('datatable', 'data'),
     Output('boxplot', 'figure')],
    Input('cohort', 'value'),
    Input('graph', 'value')
)

def update_table(cohort, column):
    filtered_df = df
    if cohort is None or cohort == []:
        pass
    else:
        filtered_df = filtered_df[filtered_df['procedure_text'].isin(cohort)]


    fig = px.box(filtered_df, x='procedure_text', y=column, title="Category Values",  points="all")
    fig.update_layout(clickmode='select', width=1100, height= 700)


    return filtered_df.to_dict('records'), fig

# @app.callback(
#     Output('selected-point', 'data'),
#     Input('boxplot', 'clickData'),
#     Input('datatable', 'data'),
# )
# def display_selected_data(clickData, data):
#     if clickData is None:
#         return "No point selected"

#     point_data = clickData['points'][0]
#     print(point_data)
#     selected_id = point_data['pointIndex']  # Retrieve customdata, which contains the ID
#     print(f'selected_id is: {selected_id}')
#     # selected_row = data.iloc[:, selected_id]
#     print(data)

#     return 'test'#f"Selected Point Data: {selected_row.to_dict('records')}"



# app.layout = html.Div([
#     html.Div(dcc.Input(id='input-on-submit', type='text')),
#     html.Button('Submit', id='submit-val', n_clicks=0),
#     html.Div(id='container-button-basic',
#              children='Enter a value and press submit')
# ])
# @callback(
#     Output('container-button-basic', 'children'),
#     Input('submit-val', 'n_clicks'),
#     State('input-on-submit', 'value'),
#     prevent_initial_call=True
# )
# def update_output(n_clicks, value):
#     return 'The input value was "{}" and the button has been clicked {} times'.format(
#         value,
#         n_clicks
#     )


if __name__ == '__main__':
    port = find_available_port()
    app.run(debug=True, port=port)
