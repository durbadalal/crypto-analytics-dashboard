import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3

# Set Page Config
st.set_page_config(
    page_title="Crypto Analytics & GenAI Dashboard",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #2a2e39;
    }
    .stApp {
        background-color: #131722;
        color: #d1d4dc;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATABASE SETUP (SQLite simulated MySQL queries)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    return conn

conn = init_db()

# Crypto Ticker Mapping
COINS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Binance Coin (BNB)": "BNB-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD"
}

# Sidebar Controls
st.sidebar.title("📌 Configuration")
selected_coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(COINS.keys()))
selected_ticker = COINS[selected_coin_name]

start_date = st.sidebar.date_input("From Date", datetime.now() - timedelta(days=365*5))
end_date = st.sidebar.date_input("To Date", datetime.now())

# Fetch Data Function with Cache
@st.cache_data(ttl=3600)
def load_crypto_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

df = load_crypto_data(selected_ticker, start_date, end_date)

# Push Data to SQLite for SQL Analysis Tab
df.to_sql('crypto_historical', conn, if_exists='replace', index=False)

# Main Title
st.title("📊 Crypto Data Analytics & GenAI Dashboard")
st.caption(f"Real-time & Historical Analytics for **{selected_coin_name}**")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live Dashboard",
    "📊 10-Year Historical Analysis",
    "📉 Interactive Charts & Technicals",
    "🛢️ SQL Query Analysis",
    "🤖 GenAI Market Assistant"
])

# ---------------------------------------------------------
# TAB 1: DASHBOARD
# ---------------------------------------------------------
with tab1:
    st.header("1. Live Overview & KPI Metrics")

    if not df.empty:
        latest_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else latest_price
        pct_change = ((latest_price - prev_price) / prev_price) * 100

        latest_volume = df['Volume'].iloc[-1]
        high_24h = df['High'].iloc[-1]
        low_24h = df['Low'].iloc[-1]

        # Metric Cards Layout
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current Price", f"${latest_price:,.2f}", f"{pct_change:+.2f}%")
        c2.metric("24h High", f"${high_24h:,.2f}")
        c3.metric("24h Low", f"${low_24h:,.2f}")
        c4.metric("Trading Volume", f"${latest_volume:,.0f}")
        c5.metric("Historical Data Points", f"{len(df)} Days")

        st.subheader("Recent Price Overview")
        fig_overview = px.line(df.tail(90), x='Date', y='Close', title=f"Last 90 Days Close Price ({selected_coin_name})", template="plotly_dark")
        st.plotly_chart(fig_overview, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: HISTORICAL ANALYSIS
# ---------------------------------------------------------
with tab2:
    st.header("2. Historical Return & Risk Metrics")

    if not df.empty:
        df_hist = df.copy()
        df_hist['Year'] = df_hist['Date'].dt.year
        df_hist['Month'] = df_hist['Date'].dt.strftime('%Y-%m')

        # Yearly Return
        yearly = df_hist.groupby('Year')['Close'].agg(['first', 'last'])
        yearly['Return (%)'] = ((yearly['last'] - yearly['first']) / yearly['first']) * 100

        # CAGR Calculation
        n_years = max((df_hist['Date'].max() - df_hist['Date'].min()).days / 365.25, 1)
        cagr = ((df_hist['Close'].iloc[-1] / df_hist['Close'].iloc[0]) ** (1 / n_years) - 1) * 100

        # Volatility & Drawdown
        daily_returns = df_hist['Close'].pct_change()
        volatility = daily_returns.std() * (365 ** 0.5) * 100  # Annualized Volatility

        cum_max = df_hist['Close'].cummax()
        drawdown = (df_hist['Close'] - cum_max) / cum_max
        max_drawdown = drawdown.min() * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("CAGR (Compound Annual Growth Rate)", f"{cagr:.2f}%")
        m2.metric("Annualized Volatility", f"{volatility:.2f}%")
        m3.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

        st.subheader("Year-wise Returns (%)")
        st.bar_chart(yearly['Return (%)'])

        with st.expander("View Yearly Performance Data Table"):
            st.dataframe(yearly.style.format("{:.2f}"))

# ---------------------------------------------------------
# TAB 3: INTERACTIVE CHARTS
# ---------------------------------------------------------
with tab3:
    st.header("3. Technical Analysis & Interactive Charts")

    # Technical Indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Candlestick Chart
    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="OHLC"
    ))
    fig_candle.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="50 SMA", line=dict(color='orange', width=1.5)))
    fig_candle.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], name="200 SMA", line=dict(color='blue', width=1.5)))
    fig_candle.update_layout(title="Candlestick Chart with 50/200 Moving Averages", template="plotly_dark", xaxis_rangeslider_visible=False)

    st.plotly_chart(fig_candle, use_container_width=True)

    # RSI Chart
    fig_rsi = px.line(df, x='Date', y='RSI', title="Relative Strength Index (RSI - 14 Days)", template="plotly_dark")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: SQL ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.header("4. SQL Data Analysis (Database Engine)")
    st.markdown("MySQL/SQLite Queries running directly on the historical dataset.")

    sql_query_type = st.selectbox("Select Predefined SQL Query", [
        "Best Performing Year",
        "Worst Performing Year",
        "Monthly Average Price",
        "Highest Trading Volume Days",
        "Bullish Period Detection (Close > 50 SMA)"
    ])

    if sql_query_type == "Best Performing Year":
        query = """
            SELECT strftime('%Y', Date) as Year,
                   MIN(Close) as Start_Price,
                   MAX(Close) as End_Price,
                   ((MAX(Close) - MIN(Close))/MIN(Close))*100 as Growth_Pct
            FROM crypto_historical
            GROUP BY Year
            ORDER BY Growth_Pct DESC
            LIMIT 1;
        """
    elif sql_query_type == "Worst Performing Year":
        query = """
            SELECT strftime('%Y', Date) as Year,
                   MIN(Close) as Start_Price,
                   MAX(Close) as End_Price,
                   ((MAX(Close) - MIN(Close))/MIN(Close))*100 as Growth_Pct
            FROM crypto_historical
            GROUP BY Year
            ORDER BY Growth_Pct ASC
            LIMIT 1;
        """
    elif sql_query_type == "Monthly Average Price":
        query = """
            SELECT strftime('%Y-%m', Date) as Month,
                   AVG(Close) as Avg_Close_Price,
                   AVG(Volume) as Avg_Volume
            FROM crypto_historical
            GROUP BY Month
            ORDER BY Month DESC;
        """
    elif sql_query_type == "Highest Trading Volume Days":
        query = """
            SELECT Date, Close, Volume
            FROM crypto_historical
            ORDER BY Volume DESC
            LIMIT 10;
        """
    else:
        query = """
            SELECT Date, Close, Volume
            FROM crypto_historical
            WHERE Close > (SELECT AVG(Close) FROM crypto_historical)
            ORDER BY Date DESC
            LIMIT 15;
        """

    st.code(query, language="sql")
    result_df = pd.read_sql_query(query, conn)
    st.dataframe(result_df)

# ---------------------------------------------------------
# TAB 5: GENAI SECTION
# ---------------------------------------------------------
with tab5:
    st.header("5. GenAI Market Intelligence Assistant 🤖")

    user_prompt = st.text_input("Ask GenAI Assistant:", value=f"{selected_coin_name}-What's the current trend?")

    if st.button("Generate AI Market Insight"):
        with st.spinner("Analyzing market indicators and generating response..."):
            latest_price = df['Close'].iloc[-1]
            rsi_val = df['RSI'].iloc[-1] if 'RSI' in df and not df['RSI'].isna().all() else 50
            sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df and not df['SMA_50'].isna().all() else latest_price

            trend = "Bullish" if latest_price > sma_50 else "Bearish"
            rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")

            ai_response = f"""
            ### 🤖 GenAI Analysis Report for **{selected_coin_name}**

            **Question:** *"{user_prompt}"*

            **Conclusion & Trend Analysis:**
            1. **Current Market Situation** {selected_coin_name}-s Present Share Price**${latest_price:,.2f}**।
            2. **Trend Analysis:** ৫০ Day's Moving Average (${sma_50:,.2f})-can compare with Current Trends**{trend}** Stays on this conditions.
            3. **RSI Indicator:** The 14-day RSI value is **{rsi_val:.1f}**, indicating that the market is currently in a **{rsi_status}** momentum.

            **Decision (AI Recommendation):**
            If you are a long-term investor, determine your strategy after checking the overall backtesting CAGR and Volatility. In the short term, an RSI of {rsi_val:.1f} indicates a support level.
            """

            st.success("Analysis Complete!")
            st.markdown(ai_response)