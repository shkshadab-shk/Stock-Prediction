
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import streamlit as st  





# Define start and end dates
start = '2010-01-01'
end = '2028-09-10'


st.title('Stock Trend Prediction')


user_input = st.text_input('Enter stock ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end , progress=False)

# Display data
st.subheader(f'Data from {start} to {end}')
st.write(df.describe())

# Display the stock price chart
st.subheader('Stock Price VS TIme Chart')
st.line_chart(df['Close'])

# Display the raw data
st.subheader('Raw Data')
st.write(df)


# 100 moving averge
if 'Close' in df.columns:
    ma100 = df['Close'].rolling(100).mean()


    # Plot the 'Close' price and the 100-day moving average
    st.subheader('Stock Price and 100-Day Moving Average')
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(ma100, label='100-Day Moving Average', color='orange')
    plt.title(f'{user_input} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)




if 'Close' in df.columns:
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    

      
    # Plot the 'Close' price, 100-day, and 200-day moving averages
    st.subheader('Stock Price, 100-Day and 200-Day Moving Averages')
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(ma100, label='100-Day Moving Average', color='orange')
    plt.plot(ma200, label='200-Day Moving Average', color='green')
    plt.title(f'{user_input} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)    

# splitting  data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler(feature_range=(0,1))
 
data_training_array = scaler.fit_transform(data_training)


# testing part
st.subheader('Orignal Price VS Predicted price')
def calculate_moving_averages(df, windows):
    ma_dict = {}
    for window in windows:
        ma_dict[f'{window}-Day MA'] = df['Close'].rolling(window=window).mean()
    return ma_dict



predictions = df['Close'].shift(-5)  # Example placeholder, shift data
predictions = predictions.fillna(method='ffill')  # Fill NaN values with previous values

# Plotting
plt.figure(figsize=(14, 7))

# Plot original close prices
plt.plot(df.index, df['Close'], label='Original Close Price', color='blue')

# Plot predicted prices
plt.plot(df.index, predictions, label='Predicted Prices', color='red')

# Labels and legend
plt.title(f'{user_input} Original vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)
