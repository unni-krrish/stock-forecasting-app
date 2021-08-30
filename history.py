from datetime import datetime as dt
from datetime import timedelta

import math
import scipy.stats as ss
import numpy as np
import pandas as pd
# plotly modules
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# Remove pandas warning
pd.options.mode.chained_assignment = None


class main_class:
    """
    Class for handling data and operations related to stock history.
    Calculates indicators, creates plotly graphs.
    """

    def __init__(self, data):
        self.data = data
        self.st_date = self.data.iloc[1, :]['Date']
        self.end_date = self.data.iloc[-1, :]['Date']
        self.int_st_date = self.data.iloc[1, :]['Date']
        self.int_end_date = self.data.iloc[-1, :]['Date']
        self.interval = self.data[(self.data['Date'] >= self.st_date) &
                                  (self.data['Date'] <= self.end_date)].reset_index(drop=True)

    def set_interval(self, start, end):
        """ 
        Resets the interval to the default. 
        Appropriate for dynamically setting the history when called from the frontend.
        """
        # Take start and end dates as inputs and filter out the fetched data with them
        self.int_st_date, self.int_end_date = start, end
        self.interval = self.data[(self.data['Date'] >= self.int_st_date) &
                                  (self.data['Date'] <= self.int_end_date)].reset_index(drop=True)

    def get_latest_value(self):
        """ Returns the latest stock closing price"""
        return self.data.iloc[-1, :]["Close"]

    def get_descriptors(self):
        """
        Gets the baisc statistical descriptors (mean, median, SD, etc.) for the
        stock history interval selected by the user.
        Returns : pandas dataframe of descriptors
        """
        # Get all the interval descriptors
        res = self.interval['Close'].describe().to_dict()
        res['c_var'] = res['std']/res['mean']
        # Create dict of all the descriptors and convert to DF
        res = {k: round(v, 2) for k, v in res.items()}
        res_out = pd.DataFrame(res, index=range(1))
        return res_out

    def get_indicators(self, req_inds, ma_window):
        """
        Takes in a list of required indicators (trend, MA, etc.) and
        calls the corresponding function to calculate the indicator for history interval.
        Returns : Dict with key = indicator code and 
                            value = pandas df with cols [Date, Indicator] 
        """
        # Define a code for all the indicators - to make inputs easier
        inds = {'linear': self.calc_linear, 'ma': self.calc_ma,
                'w_ma': self.calc_w_ma, 'e_ma': self.calc_e_ma,
                'macd': self.calc_macd, 'rsi': self.calc_rsi, 'obv': self.calc_obv}
        # Define colors and names for indicators - to be used in the history graph
        colors = {'linear': 'rgb(150,150,0)', 'ma': 'rgb(80,80,255)',
                  'w_ma': 'rgb(150,150,255)', 'e_ma': 'rgb(220,220,255)'}
        names = {'linear': 'Linear Trend', 'ma': 'Moving Average (MA)',
                 'w_ma': 'Weighted MA', 'e_ma': 'Exponential MA'}
        return {k: v(ma_window) for (k, v) in inds.items() if k in req_inds}, colors, names

    def get_expanded_interval(self, n_points):
        """
        When calculating Moving Average and RSI, the methods will need
        data before the start date of the interval. This method returns
        required extra points before the history interval start date so that
        the indicators do not start with NaN values.
        Returns : pandas df : History interval with start date pulled back by n_points
        """
        # Filter out all the data points before interval start date
        extra_pts = self.data[self.data['Date'] <
                              self.int_st_date].reset_index(drop=True)
        # Find extended start date for the new interval
        ext_st_pt = extra_pts.iloc[-n_points, :]['Date']
        # Filter out new interval using extended start date and previous end date
        strip = self.data[(self.data['Date'] >= ext_st_pt) &
                          (self.data['Date'] <= self.int_end_date)].reset_index(drop=True)
        return strip

    def calc_linear(self, window=1):
        """
        Calculates linear trendline for the stock history interval
        Returns : pandas df of interval date span and linear trend line values
        """
        # Define x and y, fit using Linear Regression in scipy stats
        y = self.interval['Close']
        x = self.interval.index
        lin_out = ss.linregress(x, y)
        m, c = lin_out[0], lin_out[1]
        return pd.concat([self.interval['Date'], pd.Series(m*x+c, name='linear')], axis=1)

    def calc_ma(self, window=8):
        """
        Calculates Moving Average for the stock history interval 
        Inputs  : user input window size for MA calculation
        Returns : pandas df of interval date span and MA values
        """
        # Get expanded interval and calculate MA by rolling mean
        strip = self.get_expanded_interval(window-1)
        ma_out = strip['Close'].rolling(window).mean()
        return pd.concat([strip['Date'], pd.Series(ma_out, name='ma')], axis=1)

    def calc_w_ma(self, window=8):
        """
        Calculates Weighted Moving Average for the stock history interval 
        Inputs  : user input window size for Weighted-MA calculation
        Returns : pandas df of interval date span and Weighted-MA values
        """
        # Get expanded interval, define weights
        strip = self.get_expanded_interval(window-1)
        weights = np.arange(1, window+1)/sum(range(1, window+1))
        # Use a dot product between rolling window and weights to calculate weighted-ma
        def wt_mean(x): return np.dot(weights, x)
        ma_out = strip['Close'].rolling(window).apply(wt_mean)
        return pd.concat([strip['Date'], pd.Series(ma_out, name='w_ma')], axis=1)

    def calc_e_ma(self, window=8):
        """
        Calculates Exponential Moving Average for the history interval 
        Inputs  : user input window size for Exponential-MA calculation
        Returns : pandas df of interval date span and Exponential-MA values
        """
        # Get expanded interval and initialise the first entry
        strip = self.get_expanded_interval(window)
        ma_out = strip['Close']
        ma_out.iloc[window-1] = ma_out.iloc[:window].mean()
        # Define exp-ma coefficient and iteratively calculate e-ma for rows
        k = 2/(window+1)
        for ix in range(window, ma_out.shape[0]):
            ma_out.iloc[ix] = ma_out.iloc[ix-1] + \
                k*(ma_out.iloc[ix]-ma_out.iloc[ix-1])
        return pd.concat([strip['Date'], pd.Series(ma_out.iloc[window:], name='e_ma')], axis=1).dropna().reset_index()

    def calc_macd(self):
        """
        Calculates Moving Average Convergence/Divergence (MACD) using the formula
        MACD = MA(window=12) - MA(window=26)
        Inputs  : window : dummy argument for get_indicators function
        Returns : pandas df of interval date span and MACD values
        """
        # Calculate exponential MA for window of 12 and 26 and take the difference
        ema_12 = self.calc_e_ma(12)
        ema_26 = self.calc_e_ma(26)
        macd = ema_12['e_ma']-ema_26['e_ma']
        return pd.concat([ema_12['Date'], pd.Series(macd, name='macd')], axis=1)

    def calc_rsi(self):
        """
        Calculates the Relative Strength Index (RSI) for the stock history interval.
        First, it uses past 14 points to get initial average gain and loss.
        Then onwards, it uses current closing price and previous RSI to get current RSI.
        RSI always uses interval of 14 points
        Returns : pandas df of interval date span and RSI values
        """
        # RSI commonly uses window of 14 days
        rsi_window = 14
        strip = self.get_expanded_interval(rsi_window+1)
        rsi = strip['Close'][rsi_window+1:].reset_index(drop=True)
        # Calculate daily changes in the last 14 days
        changes = [(rsi[ix]-rsi[ix-1])*100/rsi[ix-1]
                   for ix in range(1, rsi.shape[0])]
        # Calculate average gain and loss, update RSI for the first row in the interval
        avg_gain = np.mean([x for x in changes if x > 0])
        avg_loss = np.mean([x for x in changes if x <= 0])
        rsi[0] = 100-100/(1+avg_gain/avg_loss)

        # Iteratively update successive rows using previous RSI and current gain/loss
        rsi_int = strip['Close'][rsi_window+1:].reset_index(drop=True)
        for ix in range(1, rsi.shape[0]):
            curr_change = (rsi_int[ix]-rsi_int[ix-1])*100/rsi_int[ix-1]
            if(curr_change > 0):
                avg_gain = (avg_gain*13 + curr_change)/14
            else:
                avg_loss = (avg_loss*13 + curr_change)/14
            rsi[ix] = 100-100/(1+avg_gain/avg_loss)
        # Get the dates corresponding to calculated RSI and combine them and return the DF
        dates = strip['Date'][rsi_window+1:].reset_index(drop=True)
        return pd.concat([dates, pd.Series(rsi, name='rsi')], axis=1)

    def calc_obv(self, window):
        pass

    def plot_rsi_macd(self):
        """
        Calculates the MACD and RSI for the stock history interval and
        creates a plotly graph for the same. Graph stylised for the Dash Web App.
        Returns : plotly graph object 
        """
        # Get the MACD and RSI for the interval, merge them for plotting
        macd, rsi = self.calc_macd(), self.calc_rsi()
        df = pd.merge(macd, rsi, left_on='Date', right_on='Date')
        # x-axis should be strings so that plotly will skip weekends
        x = df['Date'].apply(lambda x: dt.strftime(x, "%Y/%m/%d"))
        # MACD-RSI plot is designed to be dual axis graph on Dash Web App
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x, y=df['rsi'],
                                 mode='lines', name='Relative Strength Index',
                                 marker_color='rgb(150,150,0)'))
        fig.add_trace(go.Scatter(x=x, y=df['macd'],
                                 mode='lines', name='MACD', marker_color='rgb(200,0,0)',),
                      secondary_y=True)
        fig.update_layout(template='plotly_dark', height=150, showlegend=True,
                          margin=dict(l=10, r=0, t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="right", x=0.6))
        fig.update_xaxes(showticklabels=False)
        return fig

    def plot_history(self, req, ma_window, mode):
        """
        Gets the user input indicators for the interval through get_indicators method.
        Plots the stock history price along with the indicators.
        Graph is stylised for Dash Web App. 
        Returns : plotly graph object  
        """
        # From the list of indicator codes, get all the calculated values
        inds, colors, names = self.get_indicators(req, ma_window)
        # Create a dataframe with date, stock prices and all the indicator values
        final = self.interval.reset_index(drop=True)
        for req_ind in inds.keys():
            final = pd.merge(final, inds[req_ind],
                             left_on='Date', right_on='Date')
        xticks = final['Date'].apply(
            lambda x: dt.strftime(x, "%Y/%m/%d"))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            horizontal_spacing=0.03, row_heights=[0.85, 0.15],
                            specs=[[{"type": "candlestick"}], [{"type": "table"}]])
        # Get statistical descriptors and create a table to add to the plotly graph
        df = self.get_descriptors()
        fig.add_trace(go.Table(
            header=dict(values=list(df.columns), align='left',
                        line_color='rgb(40,40,40)', fill_color='rgb(40,40,40)',
                        font=dict(color='rgb(180,180,180)')),
            cells=dict(values=[df[col] for col in df.columns], align='left',
                       fill_color='rgb(40,40,40)', line_color='rgb(40,40,40)',
                       font=dict(color='rgb(180,180,180)'))), row=2, col=1)
        if(mode == 'candlesticks'):
            fig.add_trace(go.Candlestick(x=xticks, name='Candlesticks',
                                         open=final['Open'], high=final['High'],
                                         low=final['Low'], close=final['Close'],
                                         increasing_line_color='rgb(40,255,40)',
                                         decreasing_line_color='rgb(255,90,90)',), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=xticks, y=final[mode],
                                     name=f"{mode} Price", marker_color='rgb(250,100,0)'), row=1, col=1)
        for ind in req:
            fig.add_trace((go.Scatter(x=xticks, y=final[ind], mode='lines',
                                      name=names[ind], marker_color=colors[ind])))
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark',
                          legend=dict(orientation="h",
                                      yanchor="bottom", y=1.02,
                                      xanchor="right", x=0.6),
                          margin=dict(l=10, r=50, t=10, b=10))
        fig.update_xaxes(nticks=20, ticks='inside',
                         tickcolor='rgb(150,150,150)', tickangle=-45)
        return fig

