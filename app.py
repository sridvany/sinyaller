import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Sayfa Konfigürasyonu
st.set_page_config(page_title="Algo-Trader Signal Pro", layout="wide")

st.title("📈 Yatırım Algortimaları Simülatörü")
ticker = st.text_input("Hisse veya Emtia Ticker Giriniz. Bilmiyorsanız ticker'ı Gemini'ye sorunuz. Bu uygulama yatırım tavsiyesi içermez. (Örn: AAPL, GC=F, BTC-USD):", "GC=F")

@st.cache_data
def get_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    return df

if ticker:
    df = get_data(ticker)
    
    if not df.empty:
        # --- HESAPLAMALAR ---
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Sinyal İşaretçileri İçin Mantık (Golden Cross / Death Cross)
        # SMA50, SMA200'ü yukarı kestiğinde +1, aşağı kestiğinde -1
        df['Signal'] = 0
        df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = 1
        df['Crossover'] = df['Signal'].diff()

        # --- PROFESYONEL GRAFİK (PLOTLY) ---
        fig = go.Figure()

        # 1. Mum Grafik (Candlestick)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Fiyat (OHLC)'
        ))

        # 2. SMA Overlay (Üst Üste Bindirme)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA 200', line=dict(color='blue', width=1.5)))

        # 3. Sinyal İşaretçileri (Oklar)
        # Al Sinyalleri (Golden Cross)
        buy_signals = df[df['Crossover'] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['SMA50'] * 0.95,
            mode='markers', name='AL Sinyali',
            marker=dict(symbol='triangle-up', size=15, color='green', line=dict(width=2, color='white'))
        ))

        # Sat Sinyalleri (Death Cross)
        sell_signals = df[df['Crossover'] == -1]
        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=sell_signals['SMA50'] * 1.05,
            mode='markers', name='SAT Sinyali',
            marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=2, color='white'))
        ))

        fig.update_layout(
            title=f"{ticker} Teknik Analiz Görünümü",
            yaxis_title="Fiyat",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- TABLO RAPORU (Önceki mantıkla aynı) ---
        st.subheader("Algoritmik Karar Destek Özeti")
        # (Buraya önceki kodundaki if/else karar bloklarını ve st.table kısmını ekleyebilirsin)
        st.info("Yukarıdaki grafik, SMA 50 ve 200 kesişimlerini (Golden/Death Cross) otomatik olarak işaretlemektedir.")
