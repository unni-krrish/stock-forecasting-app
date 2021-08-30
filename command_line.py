from datetime import datetime as dt, timedelta
import pickle
from forecast import forecaster
from fetch_stocks import get_historical
from history import main_class
inp_sep = "-"*50 + "\n"
err_sep = "\n\nERROR : "
msg_sep = "\n!!! Prompt !!!\n"


class cmd_handler:

    def __init__(self):
        self.ticker = None
        self.hist_start = None
        self.hist_end = None
        self.indicators = None
        self.ind_window = None
        self.fc_start = None
        self.fc_end = None
        self.fc_pred = None
        self.model_code = None

    def set_a_valid_date(self, prompt):
        done = False
        while not done:
            valid_out = self.validate_date(input(prompt))
            if valid_out is not None:
                return valid_out

    def set_ticker(self):
        inp = input(inp_sep + "Enter ticker symbol of the stock to fetch : ")
        done = False
        while not done:
            valid = self.validate_ticker(inp)
            if valid is not None:
                self.ticker = valid
                done = True
            else:
                print("Ticker is not valid")
                inp = input(inp_sep + "Retry entering ticker symbol : ")

    def set_hist_start(self):
        prompt = inp_sep + \
            "Enter start date for the stock history (YYYY-MM-DD) : "
        done = False
        while not done:
            valid_out = self.validate_date(input(prompt))
            if valid_out is not None and self.check_hist_start(valid_out):
                self.hist_start = valid_out
                done = True
            else:
                prompt = inp_sep + \
                    "Retry start date for the stock history (YYYY-MM-DD) : "

    def check_hist_start(self, date_in):
        left_limit = dt.strptime("2005-01-01", "%Y-%m-%d")
        right_limit = dt.today() - timedelta(days=1)
        if date_in < left_limit:
            print(err_sep + "Start Date must be later than 2005-JAN-01")
            return False
        elif date_in > right_limit:
            print(err_sep + "Start Date must sooner than Yesterday's date")
            return False
        return True

    def set_hist_end(self):
        prompt = inp_sep + \
            "Enter end date for the stock history (YYYY-MM-DD)   : "
        done = False
        while not done:
            valid_out = self.validate_date(input(prompt))
            if valid_out is not None and self.check_hist_end(valid_out):
                self.hist_end = valid_out
                done = True
            else:
                prompt = inp_sep + \
                    "Retry end date for the stock history (YYYY-MM-DD)   : "

    def check_hist_end(self, date_in):
        if date_in <= self.hist_start:
            print(err_sep + "End date comes before start date")
            return False
        return True

    def set_fc_dates(self):
        prompt = inp_sep + \
            "Enter start date for Forecast model training period (YYYY-MM-DD) : "
        self.fc_start = self.set_a_valid_date(prompt)
        prompt = inp_sep + \
            "Enter end date for Forecast model training period (YYYY-MM-DD)   : "
        self.fc_end = self.set_a_valid_date(prompt)
        prompt = inp_sep + \
            "Enter date for which prediction is to be made (YYYY-MM-DD)       : "
        self.fc_pred = self.set_a_valid_date(prompt)

    def set_model(self):
        msg = msg_sep + "\nFollowing models are available : "
        msg += "     model code    Model"
        msg += "     linear        Linear Regression"
        msg += "     arima         ARIMA"
        msg += "     ml_model      Machine Learning model (XG-Boost)"
        allowed = ['linear', 'arima', 'ml_model']
        inp = input(inp_sep +
                    "Enter model code for the Model to be used for prediction : ")
        done = False
        while not done:
            if inp.strip().lower() in allowed:
                self.model_code = inp.strip().lower()
                done = True
            else:
                inp = input(inp_sep + "Retry entering model code : ")

    def set_indicators(self):
        msg = msg_sep + "Enter required indicator codes separated by a space"
        msg += "\n    Indicator-Code    Indicator"
        msg += "\n1.  linear            Linear trend line"
        msg += "\n2.  ma                Moving Average"
        msg += "\n3.  w_ma              Weighted Moving Average"
        msg += "\n4.  e_ma              Exponential Moving Average\n"
        msg += "\nExample : 'linear ma e_ma' for Indicators 1,2 and 4\n"
        print(msg)
        done = False
        inp = input(inp_sep + "Enter indicator codes : ")
        while not done:
            res = self.check_indicators(inp)
            if res is not None:
                self.indicators = list(set(res))
                print(
                    f"\nFollowing indicators are received : {self.indicators}")
                done = True
            else:
                inp = input(inp_sep +
                            "Received no sinlge valid indicator code. Retry entering : ")

    def check_indicators(self, inp):
        allowed = ['linear', 'ma', 'w_ma', 'e_ma']
        entries = inp.strip().split()
        ind_list = []
        for ind in entries:
            if ind not in allowed:
                print(f"{ind} is not a valid entry")
                continue
            else:
                ind_list.append(ind)
        if len(ind_list) == 0:
            return None
        return ind_list

    def set_indicator_window(self):
        done = False
        inp = input(inp_sep + "Enter window for indicator calculations : ")
        while not done:
            res = self.check_indicator_window(inp)
            if res is not None:
                self.ind_window = res
                done = True
            else:
                inp = input(inp_sep +
                            "Retry entering window for indicator calculations : ")

    def check_indicator_window(self, window_in):
        try:
            window = int(window_in)
        except ValueError:
            print(err_sep + "Input contains non-numeric character(s) or space(s)")
            return None
        else:
            if window > 30:
                print(err_sep +
                      "Invalid input : Maximum supported window for Moving Averages is 30")
                return None
            else:
                return window

    def validate_date(self, date_in):
        ymd = date_in.strip().split('-')
        if len(ymd) == 3 and len(ymd[0]) == 4 and len(ymd[1]) == 2 and len(ymd[2]) == 2:
            try:
                return dt.strptime('-'.join(ymd), "%Y-%m-%d")
            except ValueError:
                print(err_sep + "Date elements contain non-numeric charater(s)\n")
        else:
            print(err_sep + "One of the date elements (Y, M, D) got unexpected length\n")

    def validate_ticker(self, tic_in):
        with open('updated_list.pkl', 'rb') as f:
            payload = pickle.load(f)
            tic_list = [x['value'] for x in payload]
        if tic_in.strip().upper() in tic_list:
            return tic_in.strip().upper()
        return None

    def get_input_summary(self):
        summary = f"\n-----------------------------------------------------\n"
        summary += f"Ticker         : {self.ticker}\n"
        summary += f"History Start  : {self.hist_start}\n"
        summary += f"History End    : {self.hist_end}\n"
        summary += f"Indicators     : {self.indicators} | Window = {self.ind_window}\n"
        summary += f"FC train start : {self.fc_start}\n"
        summary += f"FC train end   : {self.fc_end}\n"
        summary += f"FC prediction  : {self.fc_pred}\n"
        summary += f"FC Model       : {self.model_code}\n"
        summary += f"------------------------------------------------------"
        return summary

    def hist_plotter(self):
        """
        The function which connects the Dash(GUI) callback methods
        to the history module
        """
        # Pull the start_date of the history 40 days back
        # This extra data at the start is needed for Moving Averages and MACD
        dt1 = dt.strftime(self.hist_start-timedelta(days=50), "%Y-%m-%d")
        dt2 = dt.strftime(self.hist_end, "%Y-%m-%d")
        inp_data = get_historical(self.ticker, dt1, dt2).reset_index()
        if inp_data.empty:
            msg = f"Requested symbol could be unlisted/delisted from Yahoo Finance"
            return None, None, msg
        # Message for GUI to throw when the downloaded history starts after the user input start date
        if inp_data.iloc[0, :]['Date'] > self.hist_start-timedelta(days=40):
            msg = f"Requested Start Date is before Company went IPO ({inp_data.iloc[0, :]['Date']})"
            return None, None, msg
        # If there is no conflict in dates, proceed with the downloaded history
        # Create main class object - history module
        curr_stock = main_class(inp_data)
        # Set the interval from user input dates - Excluding the extra 40 days of data
        curr_stock.set_interval(self.hist_start, self.hist_end)

        # Get the plotly graph object with stock price and all the indicators requested by the user
        hist_plot = curr_stock.plot_history(
            self.indicators, self.ind_window, mode='candlesticks')
        # RSI and MACD plots are on a separate plot area - always plotted
        rsi_plot = curr_stock.plot_rsi_macd()
        return hist_plot, rsi_plot, None

    # def fc_plotter(self):
    #     fct = forecaster(inp_data, date_1, date_2, date_3)
    #     # Get the plotly graph object along with the corresponding message
    #     plot_out, prediction = fct.plot_forecast(model)


def launch_cmd():
    print("---------------     STOCK HISTORY AND FORECASTING APPLICATION     -----------------")
    cmd_pilot = cmd_handler()
    cmd_pilot.set_ticker()
    cmd_pilot.set_hist_start()
    cmd_pilot.set_hist_end()
    cmd_pilot.set_indicators()
    cmd_pilot.set_indicator_window()
    # cmd_pilot.set_model()
    print(cmd_pilot.get_input_summary())
    hist_plot, rsi_plot, msg = cmd_pilot.hist_plotter()
    if msg is None:
        hist_plot.show()
    else:
        print(f"!!!!!!!!!!!!    ERROR    !!!!!!!!!!!!\n{msg}")


if __name__ == "__main__":
    launch_cmd()
