# --------------------------
# Required Imports
# --------------------------
import yfinance as yf
import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import plotly.graph_objects as go

# --------------------------
# Load FinBERT sentiment model
# --------------------------
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# --------------------------
# Example tickers
# --------------------------
TICKERS = ["AAPL", "AMD", "AMZN", "GOOGL", "JPM", "META", "MSFT", "NFLX", "NVDA", "TSLA"]

# --------------------------
# Fake sample news (replace with scraping if needed)
# --------------------------
sample_news = {
    "AAPL": ["Apple launches new iPhone", "Apple faces supply chain issues"],
    "AMD": ["AMD reports record revenue", "AMD faces competition from Intel"],
    "AMZN": ["Amazon expands grocery delivery", "Amazon faces antitrust lawsuit"],
    "GOOGL": ["Google introduces AI tool", "Google fined in Europe"],
    "JPM": ["JP Morgan posts strong profits", "JPM faces regulatory scrutiny"],
    "META": ["Meta invests in VR", "Meta faces privacy concerns"],
    "MSFT": ["Microsoft launches new cloud service", "Microsoft faces antitrust issues"],
    "NFLX": ["Netflix adds more subscribers", "Netflix stock drops"],
    "NVDA": ["NVIDIA powers AI revolution", "NVIDIA supply shortages"],
    "TSLA": ["Tesla expands production", "Tesla recalls cars for safety issues"]
}

# --------------------------
# Functions
# --------------------------
def get_stock_data(ticker, start, end):
    """Download stock data from yfinance"""
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df

def get_sentiment(headlines):
    """Compute sentiment score using FinBERT"""
    if not headlines:
        return 0
    scores = []
    for text in headlines:
        result = sentiment_pipeline(text)[0]
        label = result["label"].lower()
        if label == "positive":
            scores.append(result["score"])
        elif label == "negative":
            scores.append(-result["score"])
        else:
            scores.append(0)
    return sum(scores) / len(scores)

# --------------------------
# Streamlit Dashboard
# --------------------------
st.set_page_config(page_title="StockPulse Dashboard", layout="wide")
st.title("📊 StockPulse: Transformer-Based NLP for Sentiment & Price Prediction")

# User selects company
selected_ticker = st.selectbox("Select a Company:", TICKERS)

# Set date range
start = datetime.date(2023, 1, 1)
end = datetime.date.today()

# Fetch stock data
df = get_stock_data(selected_ticker, start, end)

# Compute sentiment
headlines = sample_news.get(selected_ticker, [])
avg_sentiment = get_sentiment(headlines)

st.subheader(f"📰 News & Sentiment for {selected_ticker}")
st.write("Sample Headlines:")
st.write(headlines)
st.metric(label="Average Sentiment Score", value=round(avg_sentiment, 4))

# --------------------------
# Plot Stock Price
# --------------------------
st.subheader(f"📈 Stock Price Trend: {selected_ticker}")

colors = ["green" if df["Adj Close"].iloc[i] >= df["Adj Close"].iloc[i-1] else "red" for i in range(1, len(df))]
colors.insert(0, "green")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Adj Close"],
    mode='lines+markers',
    line=dict(color="lightgray"),
    marker=dict(color=colors, size=6),
    hovertemplate="Date: %{x}<br>Price: %{y}<extra></extra>"
))
fig.update_layout(
    title=f"{selected_ticker} Stock Price",
    xaxis_title="Date", yaxis_title="Price",
    template="plotly_dark",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Predictive Modeling
# --------------------------
st.subheader(f"🤖 Price Movement Prediction for {selected_ticker}")

df["Return"] = df["Adj Close"].pct_change()
df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
df["Sentiment"] = avg_sentiment
df = df.dropna()

if not df.empty:
    X = df[["Return", "Sentiment"]]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.write(f"📈 Logistic Regression Accuracy: *{acc:.2f}*")
else:
    st.write("Not enough data for prediction.")

# --------------------------
# Price Prediction with Regression
# --------------------------
st.subheader(f"💰 Price Prediction for {selected_ticker}")

df["Return"] = df["Adj Close"].pct_change()
df["MA5"] = df["Adj Close"].rolling(window=5).mean()
df["MA10"] = df["Adj Close"].rolling(window=10).mean()
df["EMA10"] = df["Adj Close"].ewm(span=10, adjust=False).mean()

# RSI calculation
window_length = 14
delta = df["Adj Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=window_length).mean()
avg_loss = pd.Series(loss).rolling(window=window_length).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

df["Sentiment"] = avg_sentiment
df = df.dropna()

if not df.empty:
    features = ["Return", "MA5", "MA10", "EMA10", "RSI", "Sentiment"]
    X = df[features]
    y = df["Adj Close"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    price_preds = reg_model.predict(X_test)
    next_price = reg_model.predict(X.iloc[-1:].values)[0]
    mse = mean_squared_error(y_test, price_preds)
    st.write(f"📉 Regression Model MSE: {mse:.2f}")
    st.write(f"🔮 Predicted Next Closing Price: *{next_price:.2f} USD*")
else:
    st.write("Not enough data for prediction.")
