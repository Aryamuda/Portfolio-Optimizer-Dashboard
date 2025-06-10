# Portfolio Optimization Dashboard

A sophisticated web application built with Streamlit and Python that empowers users to analyze and optimize investment portfolios based on Modern Portfolio Theory (MPT). This tool provides in-depth visualizations and calculates optimal asset allocations to maximize risk-adjusted returns or minimize volatility.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-optimizer-dashboard.streamlit.app/)

---

## Methodology Overview

This dashboard is built upon the principles of **Modern Portfolio Theory (MPT)**, a Nobel Prize-winning framework for asset allocation. The core idea is that an asset's risk and return should not be viewed in isolation, but by how it contributes to a portfolio's overall risk and return.

The primary goal of MPT is to construct an "efficient frontier" of portfolios. For any given level of risk, there is one portfolio that offers the highest possible return. This tool helps find specific points on that frontier based on user-defined goals. The main optimization objective is often expressed via the **Sharpe Ratio**:

$$
\text{Sharpe Ratio} = \frac{(\text{Portfolio Return} - \text{Risk-Free Rate})}{\text{Portfolio Volatility}}
$$

A higher Sharpe Ratio indicates a better risk-adjusted return. This dashboard allows users to find the portfolio with the highest possible Sharpe Ratio or the one with the absolute minimum volatility.

## Core Calculations

The optimization process relies on two fundamental statistical inputs derived from historical asset prices:

1.  **Expected Annual Returns**: The anticipated return for each asset. In this model, it is calculated as the annualized mean of historical daily returns. This serves as the "reward" component in our analysis.
2.  **Covariance Matrix**: A statistical measure of how different assets move in relation to one another. This matrix is crucial for calculating the portfolio's overall risk (volatility) and is the mathematical key to diversification. A low covariance between assets helps reduce total portfolio risk.

These two inputs—one for return, one for risk—form the foundation upon which the optimization is built, providing a quantitative basis for asset allocation decisions.

## How It Works

The entire process is orchestrated within the single Streamlit application script.

### 1. Data Fetching

Historical daily closing prices for all selected assets are downloaded from Yahoo Finance using the `yfinance` library. The data is automatically adjusted for splits and dividends to represent the **total return** series, ensuring accuracy in performance calculations.

### 2. Optimization Engine

Based on user selections, the application performs the following steps:

-   Calculates the **Expected Returns** and the **Covariance Matrix** from the historical data.
-   Initializes an `EfficientFrontier` object from the `PyPortfolioOpt` library.
-   Applies user-defined **constraints**, such as the maximum allowable weight for any single asset.
-   Runs the optimization algorithm to find the portfolio that either **maximizes the Sharpe Ratio** or **minimizes volatility**.

### 3. Visualization and Results

The calculated optimal weights and expected performance metrics (Annual Return, Volatility, Sharpe Ratio) are displayed clearly. The results are visualized using `Plotly Express` through interactive pie charts and tables for intuitive analysis.

## How to Use

### Prerequisites

The application is deployed on Streamlit Cloud. To run it locally, you will need Python 3 and the libraries listed in `requirements.txt`.

### Configuration

All key parameters can be easily modified in the application's sidebar:

-   **Select Assets**: Choose from a list of stocks and ETFs.
-   **Set Date Range**: Define the period for historical data analysis.
-   **Configure Optimization**: Select "Maximize Sharpe Ratio" or "Minimize Volatility" as your goal.
-   **Add Constraints**: Use the slider to set the maximum weight for any single asset.

### Execution

To run the app locally after installing dependencies:
```bash
streamlit run app.py
