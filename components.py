from datetime import date
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

"""
Module to create all the required HTML Web objects.
Dash App can import and use required objects.
Removes clutter in dash_app.py 
"""

# Dropdown ticker symbol selector in the UI
stock_selector = dbc.Col(dcc.Dropdown(id='select_comp', placeholder='Stock Code',
                                      multi=False,),
                         width={'size': 2, "offset": 1, },
                         )

# History start date picker
start_date = dbc.Col(dcc.DatePickerSingle(
    id='start_dt',
    date=date(2020, 1, 1),
    display_format='DD/MM/YYYY'
), width={'size': 1, 'offset': 1})

# History end date picker
end_date = dbc.Col(dcc.DatePickerSingle(
    id='end_dt',
    date=date(2020, 5, 31),
    display_format='DD/MM/YYYY'
), width={'size': 1})

# Forecast model training start date
fc_train_st = dbc.Col(dcc.DatePickerSingle(
    id='fc_start',
    date=date(2020, 1, 1),
    display_format='DD/MM/YYYY'
), width={'size': 1, "offset": 3})

# Forecast model training end date
fc_train_end = dbc.Col(dcc.DatePickerSingle(
    id='fc_end',
    date=date(2020, 5, 31),
    display_format='DD/MM/YYYY'
), width={'size': 1})

# Forecast model final prediction date
fc_pred_dt = dbc.Col(dcc.DatePickerSingle(
    id='fc_pred',
    date=date(2020, 6, 15),
    display_format='DD/MM/YYYY'
), width={'size': 1})


# Radio input field for selecting Date selection
date_span = dbc.Col(dcc.RadioItems(id='date_span',
                                   options=[{'label': 'Pick Dates', 'value': 'alt'},
                                            {'label': ' 3 mo ', 'value': 90},
                                            {'label': ' 6 mo ', 'value': 180},
                                            {'label': ' 1 yr ', 'value': 365},
                                            {'label': ' 2 yr ', 'value': 730},
                                            {'label': ' 5 yr ', 'value': 1825}],
                                   value=180,
                                   labelStyle={'display': 'inline-block'},
                                   inputStyle={'margin-left': '15px'}
                                   ), width={'size': 4})

# Dropdown multi-value selector for the Indicators in the history graph
ind_selector = dbc.Col(dcc.Dropdown(id='ind_selector', placeholder='Select Indicators',
                                    options=[{'label': 'Linear Trend', 'value': 'linear'},
                                             {'label': 'Moving Average (MA)',
                                              'value': 'ma'},
                                             {'label': 'Weighted MA',
                                              'value': 'w_ma'},
                                             {'label': 'Exponential MA', 'value': 'e_ma'}],
                                    multi=True,
                                    value=['linear', 'ma'],),
                       width={'size': 3, 'offset': 1},
                       )

# Dropdown selector to choose forecasting model
model_selector = dbc.Col(dcc.Dropdown(id='select_model', placeholder='Forecast Model',
                                      options=[{'label': 'Linear Regression', 'value': 'linear'},
                                               {'label': 'ARIMA', 'value': 'arima'},
                                               {'label': 'XG-Boost (ML)', 'value': 'ml_model'}],
                                      multi=False,
                                      value='linear',),
                         width={'size': 2},
                         )

# Button to confirm history fetching
submit_button = dbc.Col(dbc.Button("Show History", id='submit_button', color="dark", className="mr-1"),
                        width={'size': 1})

# Button to update the history graph with selected indicators
update_button = dbc.Col(dbc.Button("Update Indicators", id='update_button', color="dark", className="mr-1"),
                        width={'size': 2})

# Button to confirm forecast request from user
fc_submit = dbc.Col(dbc.Button("Forecast", id='fc_submit', color="dark", className="mr-1"),
                    width={'size': 1})

# Text field to display errors and messages related to history area
his_message = dbc.Col(html.Div(id='his_message', style={'color': 'red'}),
                      width={'size': 4, 'offset': 1})

# Text field to display errors and messages related to forecasting
fc_message = dbc.Col(html.Div(id='fc_message', style={'color': 'red'}),
                     width={'size': 4, 'offset': 0})

# History ploatting canvas
stock_plot = dbc.Col(dcc.Graph(id='chart_1', figure=go.Figure(layout=dict(template='plotly_dark'))),
                     width={'size': 6, "offset": 1}
                     )

# Canvas for RSI/MACD plot below history graph
rsi_plot = dbc.Col(dcc.Graph(id='chart_3', figure=go.Figure(layout=dict(template='plotly_dark'))),
                   width={'size': 6, "offset": 1}
                   )

# Canvas for forecast results graph
fc_plot = dbc.Col(dcc.Graph(id='chart_2', figure=go.Figure(layout=dict(template='plotly_dark'))),
                  width={'size': 5}
                  )

# Input for Moving average window
ma_window = dbc.Col(dcc.Input(
    id="ma_window",
    type='number',
    value=4,
    placeholder="input type {}".format('number'),), width={'size': 1})
