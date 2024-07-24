# MACD Trading Strategy

This repository contains a Python implementation of a trading strategy using the MACD (Moving Average Convergence Divergence) indicator. The strategy is designed to analyze historical stock data and generate buy and sell signals based on MACD crossovers.

## Overview

The MACD trading strategy aims to identify potential buy and sell opportunities by analyzing the relationship between the MACD line and the MACD signal line. The strategy uses historical stock data to generate signals, which can be used for backtesting and evaluation.

## Features

- Fetch historical stock data from Yahoo Finance.
- Calculate MACD indicators and generate buy/sell signals.
- Analyze trading performance, including net returns, buy/sell triggers, max drawdown, average profit, and average loss.

## Prerequisites

Make sure you have the following Python packages installed:

- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`
- `backtesting`

You can install these packages using `pip`:

```bash
pip install pandas numpy matplotlib yfinance backtesting
