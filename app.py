import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import streamlit as st

# Function to calculate RDC
def calculate_rdc(prices):
    # Perform linear regression
    days = np.arange(len(prices)).reshape(-1, 1)
    regression = LinearRegression()
    regression.fit(days, prices)

    # Predict prices using the regression line
    predicted_prices = regression.predict(days)

    # Calculate RDC (divergence or convergence)
    rdc = prices - predicted_prices.flatten()
    return rdc

# Function to calculate linear regression trendlines
def calculate_trendlines(prices):
    # Perform linear regression
    days = np.arange(len(prices)).reshape(-1, 1)
    regression = LinearRegression()
    regression.fit(days, prices)

    # Extract slope and intercept of the regression line
    slope = regression.coef_[0]
    intercept = regression.intercept_

    # Calculate support and resistance lines
    support_line = slope * days + intercept - np.std(prices)
    resistance_line = slope * days + intercept + np.std(prices)

    return support_line, resistance_line,slope

# Function to perform forecasting using linear regression
def linear_regression_forecast(prices, forecast_days):
    # Perform linear regression
    days = np.arange(len(prices)).reshape(-1, 1)
    regression = LinearRegression()
    regression.fit(days, prices)
    
    slope=regression.coef_[0]
    # Forecast future prices
    forecast_start =len(prices)
    forecast_end = forecast_start + forecast_days
    forecast_days_array = np.arange(forecast_start, forecast_end).reshape(-1, 1)
    forecast_prices = regression.predict(forecast_days_array)

    return forecast_prices,slope

# Function to calculate linear regression angle
def calculate_linear_regression_angle(prices):
    # Prepare X and y for linear regression
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.reshape(-1, 1)

    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X, y)

    # Calculate angle (in degrees)
    angle = np.arctan(lr.coef_[0][0]) * (180 / np.pi)
    return angle

# Function to calculate regression crossover points
def calculate_regression_crossover(prices):
    # Perform linear regression
    days = np.arange(len(prices)).reshape(-1, 1)
    regression = LinearRegression()
    regression.fit(days, prices)

    # Predict prices using the regression line
    predicted_prices = regression.predict(days)

    # Find crossover points
    crossover_points = np.where(np.diff(np.sign(prices - predicted_prices)))[0]

    return crossover_points

def analyze_stock(stock, start_date, end_date):
    # Fetch historical data of given stock
    data = yf.download(stock, start=start_date, end=end_date)

    # Extract adjusted close prices
    prices = data['Adj Close'].values

    # Calculate RDC
    rdc = calculate_rdc(prices)

    # Calculate linear regression trendlines
    support_line, resistance_line, slope_trend = calculate_trendlines(prices)

    # Number of days for forecasting
    forecast_days = 30

    # Perform linear regression and forecast future prices
    forecast_prices,slope_forecast = linear_regression_forecast(prices, forecast_days)

    # Generate date range for forecasted prices
    last_date = data.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Calculate linear regression angle
    angle = calculate_linear_regression_angle(prices)

    # Calculate regression crossover points
    crossover_points = calculate_regression_crossover(prices)

    # Plot all indicators
    plt.figure(figsize=(12, 8))

    # Plot RDC
    plt.plot(data.index, rdc, color='purple', label='RDC')
    plt.axhline(y=0,color='brown',linestyle='--',linewidth=1)

    # Plot linear regression trendlines
    plt.plot(data.index, support_line, color='green', linestyle='--', label='Support Line')
    plt.plot(data.index, resistance_line, color='red', linestyle='--', label='Resistance Line')

    # Plot historical prices
    plt.plot(data.index, prices, color='blue', label='Historical Prices')

    # Plot forecasted prices
    plt.plot(forecast_dates, forecast_prices, color='orange', label='Forecasted Prices')

    # Plot regression crossover points
    plt.plot(data.index[crossover_points], prices[crossover_points], color='black', marker='o', label='Regression Crossover')

    # Plot settings
    plt.title('Indicators for {} Stock'.format(stock))
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Print the calculated angle and crossover points
    st.write("Linear regression angle for {} stock:".format(stock), angle)
    
    # Analyze signals
    signals = {
        'Regression Angle': 'BUY' if angle > 1 else 'SELL',
        'RDC': 'BUY' if np.sum(rdc[:30] > 0) > np.sum(rdc[:30] < 0) else 'SELL',
        'Regression Crossover': 'BUY' if np.sum(prices[crossover_points[-30:]] > np.arange(len(crossover_points[-30:]))) > np.sum(prices[crossover_points[-30:]] < np.arange(len(crossover_points[-30:]))) else 'SELL',
        'Forecast': 'BUY' if slope_forecast>0 else 'SELL',
        'Trendline': 'BUY' if slope_trend > 0 else 'SELL'
    }

    st.write("Signals Analysis:")
    for signal, value in signals.items():
        st.write("{}: {}".format(signal, value))

    # Overall recommendation
    if list(signals.values()).count('BUY') > list(signals.values()).count('SELL') :
        overall_recommendation = 'BUY'
    else :
        overall_recommendation = 'SELL'
    st.write("Overall Recommendation: {}".format(overall_recommendation))


# Streamlit UI
def stock_analysis_app():
    st.title("Stock Analysis App")
    st.write("Enter the stock symbol (e.g., AAPL for Apple Inc.)")
    stock_symbol = st.text_input("Stock Symbol:")
    start_date = st.date_input("Start Date:", datetime(2022, 1, 1))
    end_date = st.date_input("End Date:", datetime(2024, 2, 28))
    
    if start_date >= end_date:
        st.error("End date must be after start date.")
        return
    
    if st.button("Analyze"):
        if stock_symbol:
            try:
                analyze_stock(stock_symbol.upper(), start_date, end_date)
            except Exception as e:
                st.error("Error occurred: {}".format(e))
                st.warning("Please make sure you entered a valid stock symbol.")

# Run the app
if __name__ == '__main__':
    stock_analysis_app()
