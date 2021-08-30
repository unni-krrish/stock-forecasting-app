from forecast import forecaster
from fetch_stocks import get_historical
from history import main_class
# HTML components
from components import fc_message, stock_selector, stock_plot, his_message
from components import submit_button, start_date, end_date
from components import fc_train_st, fc_train_end, fc_pred_dt, fc_submit, fc_plot, rsi_plot
from components import model_selector, date_span, ind_selector, update_button, ma_window

import pickle
import numpy as np
from datetime import datetime as dt
from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Dash modules
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash import callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State


def pilot(stock_code, date_1, date_2, req, ma_window, which_button):
    """
    The function which connects the Dash(GUI) callback methods
    to the history module
    """
    # If submit button is pressed, fetch historical data and display indicators
    if(which_button == 'submit_button'):
        # Pull the start_date of the history 40 days back
        # This extra data at the start is needed for Moving Averages and MACD
        dt1 = dt.strftime(date_1-timedelta(days=50), "%Y-%m-%d")
        dt2 = dt.strftime(date_2, "%Y-%m-%d")
        inp_data = get_historical(stock_code, dt1, dt2).reset_index()
        if inp_data.empty:
            msg = f"Requested symbol could be unlisted/delisted from Yahoo Finance"
            return msg, None
        # Message for GUI to throw when the downloaded history starts after the user input start date
        if inp_data.iloc[0, :]['Date'] > date_1-timedelta(days=40):
            msg = f"Requested Start Date is before Company went IPO ({inp_data.iloc[0, :]['Date']})"
            return msg, None
        # If there is no conflict in dates, proceed with the downloaded history
        # Create main class object - history module
        curr_stock = main_class(inp_data)
        # Set the interval from user input dates - Excluding the extra 40 days of data
        curr_stock.set_interval(date_1, date_2)
        # Save history object to pickle file - Updating indicators, passing data to forecast module, etc.
        # So that the app doesn't connect to yahoo finance unnecessarily
        with open('recent_object.pkl', 'wb') as f:
            pickle.dump(curr_stock, f, pickle.HIGHEST_PROTOCOL)
    # If updating indicators or forecasting, load the history object from pickle file
    else:
        with open('recent_object.pkl', 'rb') as f:
            curr_stock = pickle.load(f)
    # Get the plotly graph object with stock price and all the indicators requested by the user
    hist_plot = curr_stock.plot_history(req, ma_window, mode='candlesticks')
    # RSI and MACD plots are on a separate plot area - always plotted
    rsi_plot = curr_stock.plot_rsi_macd()
    return hist_plot, rsi_plot


# Open the ticker list so that Dash(GUI) can access it
# to dynamically update the search results in the App
with open('updated_list.pkl', 'rb') as f:
    tic_lst = pickle.load(f)

# Create Dash Web App instance
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Create App layout using elements defined in the components module
app.layout = html.Div([
    dbc.Row(html.Br()),
    dbc.Row([stock_selector, date_span, fc_pred_dt, model_selector]),
    dbc.Row(html.Br()),
    dbc.Row([start_date, end_date, submit_button,
             fc_train_st, fc_train_end, fc_submit]),
    dbc.Row(html.Br()),
    dbc.Row([ind_selector, ma_window, update_button, fc_message]),
    dbc.Row(his_message),
    dbc.Row(html.Br()),
    dbc.Row([stock_plot, fc_plot]),
    dbc.Row([rsi_plot]), ],
)


@app.callback(
    [Output('chart_1', 'figure'),
     Output('chart_3', 'figure'),
     Output('his_message', 'children'),
     Output('his_message', 'style')],
    [Input('submit_button', 'n_clicks'),
     Input('update_button', 'n_clicks')],
    [State('select_comp', 'value'),
     State('start_dt', 'date'),
     State('end_dt', 'date'),
     State('date_span', 'value'),
     State('ind_selector', 'value'),
     State('ma_window', 'value')]
)
def update_graph(button_1, button_2, stock_code, st_date, end_date, span, ind_selector, ma_window):
    """
    Controls the history plot area and associated messages displayed.
        Function will be executed only with the press of Submit or Update buttons.
        Required inputs are provided by the State callback variables.
        Data for GUI updates are returned by the function through Output callback variables.

    """
    # Identify which button is pressed by using dash callback triggers
    trig = callback_context.triggered[0]
    which_button = trig['prop_id'].split('.')[0]
    # If no buttonis pressed, provide no update, provide msg on how to proceed
    if which_button == '':
        msg = "Get History and then Update Indicators"
        style = {'color': 'yellow'}
        return dash.no_update, dash.no_update, msg, style
    # If no symbol is selected, display appropriate message
    if not stock_code:
        msg = "No valid ticker (symbol) is selected"
        style = {'color': 'red'}
        return dash.no_update, dash.no_update, msg, style
    # If user tries to update indicators without requesting a stock history,
    # Prompt a message to request history first and then change indicators
    elif which_button == 'update_button' and not button_1:
        msg = "Get History before Updating Indicators"
        style = {'color': 'red'}
        return dash.no_update, dash.no_update, msg, style
    # This branch handles date selection for fetching stock history
    # Gets executed only with the press of submit button in GUI
    else:
        # Date selection for when user manually picks start and end dates
        if span == 'alt':
            # App only supports history starting later than 2005-01-01
            left_limit = dt.strptime("2005-01-01", "%Y-%m-%d")
            right_limit = dt.today() - timedelta(days=1)
            date_1 = dt.strptime(st_date, "%Y-%m-%d")
            date_2 = dt.strptime(end_date, "%Y-%m-%d")
            # Prompt user to pick a start date later than the limit
            if date_1 < left_limit:
                msg = "Start Date must be later than 2005-JAN-01"
                style = {'color': 'red'}
                return dash.no_update, dash.no_update, msg, style
            if date_1 > right_limit:
                msg = "Start Date must sooner than Yesterday's date"
                style = {'color': 'red'}
                return dash.no_update, dash.no_update, msg, style
        # Date selection for when user picks a date span
        # In this case, end date is fixed on today's date
        else:
            date_2 = dt.today()
            date_1 = date_2-timedelta(days=span)
    if date_1 >= date_2:
        msg = "Invalid input : Start Date comes after End date"
        style = {'color': 'red'}
        return dash.no_update, dash.no_update, msg, style
    if ma_window > 30:
        msg = "Invalid input : Maximum supported window for Moving Averages is 30"
        style = {'color': 'red'}
        return dash.no_update, dash.no_update, msg, style
    # Once the dates are set, execute the pilot function to get history graph
    # and RSI/MACD graph (as plotly graph objects) with all statistical descriptors
    plot_out, rsi_plot = pilot(
        stock_code, date_1, date_2, ind_selector, ma_window, which_button)
    # When the pilot function return None object, it means that the user
    # has requested stock price for a date before the company went IPO
    # Here, plot_out is a message to be displayed to prompt the user
    if not rsi_plot:
        style = {'color': 'red'}
        return dash.no_update, dash.no_update, plot_out, style
    # When user makes a completely successful request, both graphs are returned with success message
    msg = "Update Indicators Dynamically"
    style = {'color': 'green'}
    return plot_out, rsi_plot, msg, style


@app.callback(
    [Output('chart_2', 'figure'),
     Output('fc_message', 'children'),
     Output('fc_message', 'style')],
    [Input('fc_submit', 'n_clicks')],
    [State('fc_start', 'date'),
     State('fc_end', 'date'),
     State('fc_pred', 'date'),
     State('select_model', 'value'),
     State('submit_button', 'n_clicks')]
)
def update_forecast(button, fc_st, fc_end, fc_pred, model, submit_button):
    """
    Updates the forecast graph in the App
    Function is executed only on the press of forecast button
    Takes in inputs through callback State variables : forecast dates, model
    Returns : Forecast graph as plotly graph object, message to prompt the user
    """
    # Prevents auto callbacks on the first loading
    if(not button):
        style = {'color': 'yellow'}
        return dash.no_update, "Get a History before forecasting", style
    # If history-submit button is not pressed, forecast will not work
    # Forecasting makes use of the already fetched history
    elif(not submit_button):
        style = {'color': 'red'}
        return dash.no_update, "Get a History before forecasting", style
    # Process the forecasting date, model training start and end dates
    else:
        date_1 = dt.strptime(fc_st, "%Y-%m-%d")
        date_2 = dt.strptime(fc_end, "%Y-%m-%d")
        date_3 = dt.strptime(fc_pred, "%Y-%m-%d")
        # Make sure that prediction date comes after model training end date
        if date_3 <= date_2:
            msg = "Prediction end date must be later than training end date"
            style = {'color': 'red'}
            return dash.no_update, msg, style
        if date_2 <= date_1:
            msg = "Training end date must be later than training start date"
            style = {'color': 'red'}
            return dash.no_update, msg, style
        # Load the current history object to get stock prices for forecasting
        with open('recent_object.pkl', 'rb') as f:
            curr_stock = pickle.load(f)
        inp_data = curr_stock.data
    if date_1 < curr_stock.int_st_date or date_2 > curr_stock.int_end_date:
        msg = "Model training period must fall in the stock price history period"
        style = {'color': 'red'}
        return dash.no_update, msg, style
    if np.busday_count(fc_st, fc_end) < 20 and model == 'ml_model':
        msg = "There should be at least 20 days in the training period for XG-Boost"
        style = {'color': 'red'}
        return dash.no_update, msg, style
    # pass the input data to the forecast module along with dates
    fct = forecaster(inp_data, date_1, date_2, date_3)
    # Get the plotly graph object along with the corresponding message
    plot_out, prediction = fct.plot_forecast(model)
    style = {'color': 'green'}
    return plot_out, f"Prediction for requested date is {prediction:.2f}", style


@app.callback(
    [Output('start_dt', 'disabled'),
     Output('end_dt', 'disabled')],
    [Input('date_span', 'value')]
)
def update_date_selection(date_span):
    """
    Disables date pickers in history section until user activates it
    using the "Pick Dates" radio button
    """
    if(date_span == 'alt'):
        return [False, False]
    else:
        return [True, True]


@app.callback(
    dash.dependencies.Output("select_comp", "options"),
    [dash.dependencies.Input("select_comp", "search_value")],
)
def update_options(query):
    """
    Dynamically updates the ticker symbol search box.
    Returns : list of available tickers with a short description.
    """
    # Raise PreventUpdate when no query is made.
    # Useful at first time loading and refreshing.
    if not query:
        raise PreventUpdate
    return [tic for tic in tic_lst if query.upper() in tic["label"]]


if __name__ == '__main__':
    app.run_server(debug=True)
