import yfinance as yf
import pandas as pd
from datetime import datetime

# Define the ticker symbol for S&P 500
ticker_symbol = "^GSPC"

# Define the start and end dates
start_date = (datetime.now().year - 10, 1, 1)  # 10 years ago from current year
end_date = datetime.now().strftime("%Y-%m-%d")  # Current date

# Fetch the historical data for the S&P 500
sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Select relevant columns: Open, High, Low, Close, Volume
sp500_data['Turnover'] = sp500_data['Volume'] * sp500_data['Close']  # Calculate Turnover
sp500_data = sp500_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']]

# Save to CSV
sp500_data.to_csv("SP500_10_years_data.csv")
