import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt


ticker = "BHEL.NS"  
df = yf.download(ticker, start="2023-01-01", end="2024-01-01", interval="1d")


macd = ta.macd(df['Close'])
df = pd.concat([df, macd], axis=1)  


df['Buy_Signal'] = np.where(
    (df['MACD_12_26_9'] > df['MACDs_12_26_9']) & (df['MACD_12_26_9'].shift(1) <= df['MACDs_12_26_9'].shift(1)), 1, 0
)
df['Sell_Signal'] = np.where(
    (df['MACD_12_26_9'] < df['MACDs_12_26_9']) & (df['MACD_12_26_9'].shift(1) >= df['MACDs_12_26_9'].shift(1)), -1, 0
)


df['Signal'] = df['Buy_Signal'] + df['Sell_Signal']


df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')  
df['Daily_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']
df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod() - 1


initial_investment = 100000  
df['Net_Return'] = initial_investment * (1 + df['Cumulative_Return'])


buy_triggers = df['Buy_Signal'].sum()
sell_triggers = df['Sell_Signal'].sum()


df['Peak'] = df['Net_Return'].cummax()
df['Drawdown'] = df['Net_Return'] - df['Peak']
df['Drawdown'] = df['Drawdown'].clip(lower=0)
max_drawdown = df['Drawdown'].max()


trades = df[df['Signal'] != 0]
trades['Trade_Return'] = trades['Net_Return'].shift(-1) - trades['Net_Return']
profits = trades[trades['Trade_Return'] > 0]['Trade_Return']
losses = trades[trades['Trade_Return'] < 0]['Trade_Return']
average_profit = profits.mean() if not profits.empty else 0
average_loss = losses.mean() if not losses.empty else 0


print(f"Initial Investment: ${initial_investment}")
print(f"Final Value of Investment: ${df['Net_Return'].iloc[-1]:.2f}")
print(f"Total Return: {df['Cumulative_Return'].iloc[-1] * 100:.2f}%")
print(f"Number of Buy Triggers: {buy_triggers}")
print(f"Number of Sell Triggers: {sell_triggers}")
print(f"Maximum Drawdown: {max_drawdown:.2f}")
print(f"Average Profit per Trade: {average_profit:.2f}")
print(f"Average Loss per Trade: {average_loss:.2f}")


plt.figure(figsize=(14, 7))


plt.subplot(4, 1, 1)
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.title('Nifty 50 Daily Prices')
plt.legend()


plt.subplot(4, 1, 2)
plt.plot(df.index, df['MACD_12_26_9'], label='MACD', color='blue')
plt.plot(df.index, df['MACDs_12_26_9'], label='MACD Signal', color='red')
plt.bar(df.index, df['MACDh_12_26_9'], label='MACD Histogram', color='grey', alpha=0.3)
plt.title('MACD Indicator')
plt.legend()


plt.subplot(4, 1, 3)
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.scatter(df.index[df['Buy_Signal'] == 1], df['Close'][df['Buy_Signal'] == 1], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(df.index[df['Sell_Signal'] == -1], df['Close'][df['Sell_Signal'] == -1], marker='v', color='red', label='Sell Signal', alpha=1)
plt.title('Buy and Sell Signals')
plt.legend()


plt.subplot(4, 1, 4)
plt.plot(df.index, df['Net_Return'], label='Net Return', color='purple')
plt.title('Net Return')
plt.legend()

plt.tight_layout()
plt.show()
