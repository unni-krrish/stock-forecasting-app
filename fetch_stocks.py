import ftplib
import yfinance as yf
import pickle
import os

curr_path = os.path.abspath('')


def download_file(filename):
    """
    Connects to NASDAQ trader FTP and downloads the files with
    the info about listed stocks, saves them to local directory. 
    """
    # FTP directory details
    path = 'SymbolDirectory/'
    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    # Login as anonymous - public FTP
    ftp.login()
    ftp.cwd(path)
    ftp.retrbinary("RETR " + filename, open(filename, 'wb').write)
    ftp.quit()


def update_list(filenames):
    """
    Creates a list of ticker symbols from the files 
    downloaded from NASDAQ FTP. 
    The file is maintained in a predefined format. Processing the text
    row by row will yield the list of traded companies.
    """
    lst = []
    # Iterate over the filenames provided
    for i_file in filenames:
        download_file(i_file)
        with open(i_file, 'r') as f:
            cont = f.readlines()
            # Iterate over the text row by row
            for i in range(1, len(cont)):
                curr = cont[i].split('|')
                tic, des = curr[0], curr[1]
                # Save date to a dict for the ticker and attach to the list
                ret = {}
                ret['label'] = tic + ' - ' + des
                ret['value'] = tic
                lst.append(ret)
    # Dump the processed list of companies into the pickle file
    # Pickle format is used for speed and efficiency reasons
    with open(os.path.join(curr_path, 'updated_list.pkl'), 'wb') as f:
        pickle.dump(lst, f, pickle.HIGHEST_PROTOCOL)


def get_historical(sym, dt1, dt2):
    """
    Fetches the stock price history between two dates
    for the requested symbol.
    Returns : pandas dataframe : history for the ticker symbol input 
    """
    # Create ticker instance and download history df
    tic = yf.Ticker(sym)
    hist = tic.history(start=dt1, end=dt2)
    return hist


# update_list(['nasdaqlisted.txt', 'otherlisted.txt'])
