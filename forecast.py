from datetime import datetime as dt
from datetime import timedelta
import math
import scipy.stats as ss
import numpy as np
import pandas as pd
# models
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# plotly modules
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class forecaster:
    """
    Class that handles the forecasting. Need data from main_class object
    in history module to train a model and make predictions. 
    Also calculates the metrics of the training models.
    Makes a plotly plot that shows training prediction and forecast.
    """

    def __init__(self, data, tr_st_date, tr_end_date, pred_date):
        self.data = data
        self.tr_st_date = tr_st_date
        self.tr_end_date = tr_end_date
        self.pred_date = pred_date

        self.train_int = self.data[(self.data['Date'] >= self.tr_st_date) &
                                   (self.data['Date'] <= self.tr_end_date)]
        self.train_int.reset_index(drop=True, inplace=True)
        self.pred_int = pd.bdate_range(self.tr_end_date, self.pred_date)
        self.pred_int = self.pred_int.to_series(
        ).reset_index(drop=True)

    def get_linear_data(self, window=4):
        """
        Creates the features and labels for Linear Regression
        and ARIMA models. Makes use of the preset training interval and 
        prediction interval which are defined by the user input dates.
        Returns : feature data for training (x_train), training labels (y_train),
                  feature data for prediction interval (x_pred)
        """
        # label is the stock closing price
        y_train = self.train_int['Close']
        x_train = np.arange(y_train.shape[0]).reshape(-1, 1)
        # Calculate train and prediction lengths to generate x_pred
        train_len, pred_len = y_train.shape[0], self.pred_int.shape[0]
        x_pred = np.arange(train_len, train_len + pred_len).reshape(-1, 1)
        return x_train, y_train, x_pred

    def linear_model(self):
        """
        Gets training data from get_linear_data method and
        uses Linear Regression model in sci-kit learn to train
        and make predctions.
        Returns : predictions for training interval (y_valid),
                  forecast values for the prediction interval 
        """
        # Get the data and fit LR model
        x_train, y_train, x_pred = self.get_linear_data()
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_valid = model.predict(x_train)
        # Make predictions
        y_pred = model.predict(x_pred)
        # Return the results in the standard format followed by other models as well
        return {'name': 'Linear Regression', 'y_valid': y_valid, 'pred': y_pred}

    def arima_model(self):
        """
        Gathers relevant data from get_linear_data method and
        uses ARIMA model in statsmodels module to train and forecast.
        Returns : predictions for training interval (y_valid),
                  forecast values for the prediction interval 
        """
        # Get the data and fit ARIMA model
        _, y_train, x_pred = self.get_linear_data()
        model = ARIMA(y_train, order=(3, 0, 3))
        fitted = model.fit()
        # Get model predictions for the training interval
        y_valid = fitted.fittedvalues
        # Make predictions
        y_pred = fitted.predict(
            start=y_train.shape[0], end=y_train.shape[0] + x_pred.shape[0])
        # Adjust for indexing correction
        y_valid[0] = y_valid[1]
        # Return the results in the standard format
        return {'name': 'ARIMA Model', 'y_valid': y_valid, 'pred': y_pred}

    def get_ml_data(self, window=4):
        """
        Creates dataset for Machine Learning models.
        Returns : feature data for training (xx_train), training labels (yy_train),
                  feature data for prediction interval (x_pred)
        """
        y_train = self.train_int['Close'].values
        # Training features are lag values of stock price.
        xx_train = np.zeros((y_train.shape[0]-window-1, window))
        # Training labels are stock prices - alike Linear regression
        yy_train = np.zeros(y_train.shape[0]-window-1)
        xx_pred = y_train[-window:].reshape(1, -1)
        # Iteratively update the training features and labels
        for i in range(xx_train.shape[0]):
            xx_train[i] = y_train[i:i+window].reshape(1, -1)
            yy_train[i] = y_train[i+window]
        return xx_train, yy_train, xx_pred

    def ml_model(self, window=4):
        """
        Gathers relevant data from get_ml_data method and
        uses Gradient Boosting model in from sk-learn module to train and forecast.
        Returns : predictions for training interval (y_valid),
                  forecast values for the prediction interval 
        """
        # Get ML training data and fit the model
        x_train, y_train, x_pred = self.get_ml_data(window=window)
        model = XGBRegressor(n_estimators=100)
        model.fit(x_train, y_train)
        # Make the first prediction
        y_valid = model.predict(x_train)
        y_pred = np.zeros(self.pred_int.shape[0])
        # Iteratively make further predictions
        # Need iterative method as future predictions depend on preceding prediction points
        for i in range(y_pred.shape[0]):
            # Prediction for ith point in future
            y_pred[i] = model.predict(x_pred[-1:, :])
            tmp = np.append(x_pred[-1], y_pred[i])
            # Update features (lag values) for the (i+1)th prediction
            x_pred = np.concatenate([x_pred, tmp[1:].reshape(1, -1)])
        # No data is available for the ML model at the start of training,
        # so appoximate with the closest point (first element of y_valid)
        y_valid = np.append(y_valid[0]*np.ones(window+1), y_valid)
        return {'name': 'Grad-Boost Regressor', 'y_valid': y_valid, 'pred': y_pred}

    def calc_metrics(self, y_valid):
        """
        Calculates following metrics for any of the above used models.
            RMSE - Root Mean Squared Error
            MAE  - Mean Absolute Error
            R2   - R-squared score
        Returns : pandas dataframe with metric names as columns and their values as first row
        """
        y_train = self.train_int['Close']
        window = y_train.shape[0]
        # Calculate all the metrics using methods in sklearn.metrics
        rmse = np.sqrt(mean_squared_error(y_train, y_valid))
        mae = mean_absolute_error(y_train, y_valid)
        r2 = r2_score(y_train, y_valid)
        # Create dict and convert to pandas df
        res = {'Prediction Window': f"{window} days",
               'RMSE': f"{rmse:.2f}", 'MAE': f"{mae:.2f}", 'R2-Score': f"{r2:.3f}"}
        df = pd.DataFrame(res, index=range(1))
        return df

    def plot_forecast(self, input_model):
        """
        Create a plotly graph object to display the following:
            - Actual stock closing price in model training interval
            - Predicted (Fitted) closing price in the training interval
            - Forecast closing price in the entire prediction interval
        Executes the corresponding method for the requested model to get the data for plotting.
        Returns : plotly graph object with the above said items
        """
        # Dict that interprets the model code input by the user from GUI
        model_selector = {'linear': self.linear_model, 'arima': self.arima_model,
                          'ml_model': self.ml_model}
        # Directly execute the model method and get all the prediction data
        name, y_valid, y_pred = model_selector[input_model]().values()
        # Calculate metrics for training interval using training predictions
        df = self.calc_metrics(y_valid)
        # There are two parts for the X-axis in the graph - Training period and Forecast period
        # Define separate x-axis for both
        x1 = self.train_int['Date'].apply(lambda x: dt.strftime(x, "%Y/%m/%d"))
        x2 = self.pred_int.apply(lambda x: dt.strftime(x, "%b-%d"))
        # Subplots for graph and the table containing metrics
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            horizontal_spacing=0.03, row_heights=[0.85, 0.15],
                            specs=[[{"type": "xy"}], [{"type": "table"}]])
        fig.add_trace(go.Scatter(x=x1, y=self.train_int['Close'],
                                 mode='lines', name='Original',
                                 marker_color='rgb(200,200,200)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x1, y=y_valid,
                                 mode='lines', name='Training Fit',
                                 marker_color='rgb(160,120,0)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x2, y=y_pred,
                                 mode='lines', name='Forecast',
                                 marker_color='rgb(220,160,0)'), row=1, col=1)
        fig.add_trace(go.Table(header=dict(values=list(df.columns), align='left',
                                           line_color='rgb(40,40,40)', fill_color='rgb(40,40,40)',
                                           font=dict(color='rgb(180,180,180)')),
                               cells=dict(values=[df[col] for col in df.columns], align='left',
                                          fill_color='rgb(40,40,40)', line_color='rgb(40,40,40)',
                                          font=dict(color='rgb(180,180,180)'))), row=2, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark',
                          legend=dict(orientation="h",
                                      yanchor="bottom", y=1.02,
                                      xanchor="right", x=0.6),
                          margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(nticks=12, ticks='inside',
                         tickcolor='rgb(150,150,150)', tickangle=-45)
        # Final forecast value for the prediction date requested by the user
        prediction = np.array(y_pred)[-1]
        # Return graph object and the final predicted closing price
        return fig, prediction
