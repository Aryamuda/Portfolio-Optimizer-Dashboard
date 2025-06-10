import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import date
import time

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# TICKER LISTS
AVAILABLE_TICKERS = sorted([
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "JNJ",
    "XOM", "PG", "BAC", "KO", "CVX", "SPY", "QQQ", "IWM", "DIA", "EFA",
    "EEM", "VWO", "TLT", "GLD", "SLV", "VNQ", "XLK", "XLF", "XLE",
    "XLV", "XLI", "ARKK", "ARKG"
])
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "SPY", "NVDA"]


# HELPER FUNCTIONS
@st.cache_data
def get_ticker_descriptions(tickers):
    descriptions = {}
    progress_bar = st.progress(0, text="Fetching company names...")
    for i, ticker_str in enumerate(tickers):
        try:
            ticker_obj = yf.Ticker(ticker_str)
            name = ticker_obj.info.get('longName', ticker_obj.info.get('shortName', ticker_str))
            descriptions[ticker_str] = f"{ticker_str} ({name})"
        except Exception:
            descriptions[ticker_str] = ticker_str
        progress_bar.progress((i + 1) / len(tickers))
    progress_bar.empty()
    return descriptions


@st.cache_data
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
    if len(tickers) == 1: data = data.to_frame(tickers[0])
    if data.empty: return pd.DataFrame(), pd.DataFrame()
    data.dropna(axis=1, how='all', inplace=True)
    log_returns = np.log(data / data.shift(1)).dropna()
    return data, log_returns


def run_optimization(price_data, optimization_goal="max_sharpe", max_weight=1.0):
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    if optimization_goal == "max_sharpe":
        ef.max_sharpe()
    elif optimization_goal == "min_volatility":
        ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)
    return cleaned_weights, performance


# UI & APP SETUP
ticker_descriptions = get_ticker_descriptions(AVAILABLE_TICKERS)

# SIDEBAR
with st.sidebar:
    st.title("Controls")
    st.markdown("---")

    # Collapsible Asset Selector
    with st.expander("1. Select Assets", expanded=False):
        select_all = st.toggle("Select All Assets", value=False)
        if select_all:
            tickers = AVAILABLE_TICKERS
            st.multiselect("Choose your assets", options=AVAILABLE_TICKERS, default=AVAILABLE_TICKERS,
                           format_func=lambda ticker: ticker_descriptions.get(ticker, ticker), disabled=True)
        else:
            tickers = st.multiselect("Choose your assets", options=AVAILABLE_TICKERS, default=DEFAULT_TICKERS,
                                     format_func=lambda ticker: ticker_descriptions.get(ticker, ticker))

    st.header("2. Set Date Range")
    date_col1, date_col2 = st.columns(2)
    start_date = date_col1.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = date_col2.date_input("End Date", date.today())

    st.header("3. Configure Optimization")
    optimization_goal = st.selectbox("Optimization Goal", ["Maximize Sharpe Ratio", "Minimize Volatility"])
    selected_opt_goal = "max_sharpe" if optimization_goal == "Maximize Sharpe Ratio" else "min_volatility"

    st.header("4. Add Constraints")
    max_weight_pct = st.slider("Max Weight per Asset (%)", 5, 100, 30, 5,
                               help="Set the maximum percentage any single asset can hold. 100% means no constraint.")
    max_weight_decimal = max_weight_pct / 100.0

    st.markdown("---")
    run_button = st.button("Run Analysis & Optimization", type="primary", use_container_width=True)

# MAIN APPLICATION
st.title("Portfolio Optimization Dashboard")

with st.expander("What is this tool?"):
    st.markdown("""
    This dashboard helps you build an optimal investment portfolio based on **Modern Portfolio Theory (MPT)**.

    MPT is a Nobel Prize-winning theory showing how to maximize portfolio expected return for a given amount of risk. The key insight is that an asset's risk and return characteristics should not be viewed alone, but by how they contribute to the portfolio's overall risk-adjusted **total return**.

    **How to use this tool:**
    1.  **Select assets** in the sidebar.
    2.  Choose your **optimization goal**: Maximize the risk-adjusted return (Sharpe Ratio) or build the safest portfolio (Minimize Volatility).
    3.  Set **constraints** to enforce diversification.
    4.  Click **'Run Analysis'** to see the results.
    """)

if not tickers:
    st.warning("Please select at least one asset from the 'Select Assets' section in the sidebar to begin.")

if run_button and tickers:
    with st.spinner("Fetching data and running analysis..."):
        price_data, log_returns = fetch_data(tickers, start_date, end_date)

        if price_data.empty:
            st.error(
                f"Could not fetch data for the selected tickers. Please check the tickers or adjust the date range.")
        else:
            tabs = st.tabs(["Data & Correlations", "Performance Analysis", "Optimized Portfolio"])

            with tabs[0]:
                st.header("Sample of Historical Price Data")
                st.markdown(
                    "This table shows a sample of the first 5 days of historical prices.")

                # Show only top 5 rows of data
                st.dataframe(price_data.head().style.format("{:.2f}"))

                st.header("Asset Correlation Heatmap")
                corr_matrix = log_returns.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                     title="Correlation Matrix of Log Returns", color_continuous_scale='RdYlGn',
                                     zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)
                with st.expander("How to Interpret the Correlation Heatmap"):
                    st.markdown("""
                    This heatmap shows how asset prices move in relation to each other, with values from -1 to +1:
                    - **+1 :** Perfect Positive Correlation. The assets tend to move together.
                    - **-1 :** Perfect Negative Correlation. The assets tend to move in opposite directions. This is excellent for diversification.
                    - **0 :** No Correlation. The assets' movements are random relative to each other.

                    **Why it matters:** Diversification aims to combine assets with low or negative correlation to reduce overall portfolio risk.
                    """)

            with tabs[1]:
                st.header("Cumulative Returns")
                cumulative_returns = (1 + log_returns).cumprod()
                fig_cum_returns = px.line(cumulative_returns, title="Growth of $1 Investment")
                st.plotly_chart(fig_cum_returns, use_container_width=True)
                with st.expander("About Cumulative Returns"):
                    st.markdown(
                        "This chart shows the total growth of a $1 investment in each asset over time, including reinvested dividends. It helps visualize historical performance.")

                st.header("Risk vs. Return")
                risk_return_df = pd.DataFrame({'Mean Annual Return': log_returns.mean() * 252,
                                               'Annual Volatility': log_returns.std() * np.sqrt(252)},
                                              index=log_returns.columns)
                fig_risk_return = px.scatter(risk_return_df, x='Annual Volatility', y='Mean Annual Return',
                                             text=risk_return_df.index,
                                             title="Mean Annual Return vs. Annual Volatility (Risk)")
                st.plotly_chart(fig_risk_return, use_container_width=True)
                with st.expander("Understanding the Risk vs. Return Plot"):
                    st.markdown("""
                    This plot visualizes the risk-reward trade-off:
                    - **Y-Axis (Return):** Average annual growth. Higher is better.
                    - **X-Axis (Volatility):** Price fluctuation (risk). Lower is better.

                    The ideal asset is in the **top-left**: high return, low risk.
                    """)

            with tabs[2]:
                st.header("Optimized Portfolio")

                try:
                    weights, performance = run_optimization(price_data, selected_opt_goal, max_weight_decimal)
                    weights = {asset: weight for asset, weight in weights.items() if weight > 1e-5}

                    st.subheader(f"Portfolio Optimized for: {optimization_goal}")
                    st.info(f"No single asset exceeds {max_weight_pct}% of the portfolio.")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Annual Return", f"{performance[0]:.2%}")
                    col2.metric("Annual Volatility (Risk)", f"{performance[1]:.2%}")
                    col3.metric("Sharpe Ratio", f"{performance[2]:.2f}")

                    with st.expander("What do these metrics mean?"):
                        st.markdown("""
                        - **Expected Annual Return:** The portfolio's predicted average yearly gain based on historical total returns.
                        - **Annual Volatility:** The predicted range of price swings (risk).
                        - **Sharpe Ratio:** The score for risk-adjusted return. A higher Sharpe Ratio is generally better.
                        """)

                    st.markdown("---")
                    st.subheader("Optimal Asset Allocation")
                    weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])

                    if not weights_df.empty:
                        col4, col5 = st.columns([0.6, 0.4])
                        with col4:
                            fig_pie = px.pie(weights_df, names='Asset', values='Weight',
                                             title="Optimized Portfolio Weights")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col5:
                            st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}), use_container_width=True,
                                         hide_index=True)

                        csv = weights_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Weights as CSV", data=csv, file_name='optimized_weights.csv',
                                           mime='text/csv', use_container_width=True)
                    else:
                        st.warning(
                            "No assets were allocated weight based on the optimization criteria. This can happen with very tight constraints or unusual market data.")

                except Exception as e:
                    st.error(f"An error occurred during optimization: {e}")
                    st.warning(
                        "This can happen if the date range is too short or if constraints are too tight for a solution to be found. Try expanding the date range or increasing the max weight per asset.")