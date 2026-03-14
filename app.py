import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# 1. SAYFA KONFİGÜRASYONU
st.set_page_config(page_title="Algo-Trader Pro v3.0", layout="wide")

# 2. OTOMATİK YENİLEME (60 Saniyede bir sayfayı günceller)
st_autorefresh(interval=60 * 1000, key="terminal_refresh")

st.title("📈 Yatırım Algoritmaları Terminali")
st.caption("Piyasa verileri Yahoo Finance üzerinden 1 dakika gecikmeli/canlı olarak çekilmektedir.")

# 3. YAN PANEL (SIDEBAR) - KONTROL MERKEZİ
with st.sidebar:
    st.header("⚙️ Veri & Algoritma Ayarları")
    ticker = st.text_input("Ticker Sembolü:", "GC=F")
    
    # PERİYOT SEÇİMİ
    period = st.selectbox(
        "Toplam Veri Süresi (Period):",
        options=["1d", "5d", "1mo", "6mo", "1y", "5y", "max"],
        index=1  # Varsayılan: 5d (1 dakikayı görmek için ideal)
    )
    
    # ARALIK (INTERVAL) SEÇİMİ - 1 DAKİKA BURADA!
    # Yahoo Kuralları: 1m verisi sadece 1d veya 5d periyotlarında çalışır.
    if period in ["1d", "5d"]:
        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"]
        default_int_idx = 0 # Varsayılan: 1m (İstediğin seçenek)
    elif period == "1mo":
        interval_options = ["2m", "5m", "15m", "30m", "60m", "1h", "1d"]
        default_int_idx = 6 # 1d
    else:
        interval_options = ["1h", "1d", "1wk", "1mo"]
        default_int_idx = 1 # 1d
        
    interval = st.selectbox("Mum Aralığı (Interval):", options=interval_options, index=default_int_idx)

    st.write("---")
    st.subheader("Algoritma Hassasiyeti")
    sma_s = st.slider("Hızlı SMA (Kısa):", 5, 50, 20)
    sma_l = st.slider("Yavaş SMA (Uzun):", 50, 200, 100)
    
    st.write("---")
    st.info("İpucu: 1 dakikalık analizler için Periyot: 5d, Mum Aralığı: 1m seçiniz.")

# 4. VERİ ÇEKME MOTORU
@st.cache_data(ttl=60)
def fetch_live_data(symbol, p, i):
    try:
        data = yf.download(symbol, period=p, interval=i)
        return data
    except Exception as e:
        st.error(f"Hata: {e}")
        return pd.DataFrame()

if ticker:
    df = fetch_live_data(ticker, period, interval)
    
    if not df.empty:
        # Sütun Düzeltme (MultiIndex Fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df['Close'].squeeze()
        
        # --- 5. ALGORİTMA HESAPLAMALARI ---
        df['SMA_SHORT'] = close.rolling(window=sma_s).mean()
        df['SMA_LONG'] = close.rolling(window=sma_l).mean()
        df['Sig'] = 0
        df.loc[df['SMA_SHORT'] > df['SMA_LONG'], 'Sig'] = 1
        df['Cross'] = df['Sig'].diff()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger
        df['Mid'] = close.rolling(window=20).mean()
        df['Std'] = close.rolling(window=20).std()
        df['Up'] = df['Mid'] + (df['Std'] * 2)
        df['Low'] = df['Mid'] - (df['Std'] * 2)

        # MACD
        e12 = close.ewm(span=12, adjust=False).mean()
        e26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = e12 - e26
        df['MACD_S'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Z-Score
        df['Z'] = (close - close.rolling(30).mean()) / close.rolling(30).std()

        # --- 6. GRAFİK TASARIMI ---
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_SHORT'], name=f'SMA {sma_s}', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_LONG'], name=f'SMA {sma_l}', line=dict(color='cyan')))

        # Oklar
        buys = df[df['Cross'] == 1]
        fig.add_trace(go.Scatter(x=buys.index, y=buys['SMA_SHORT']*0.99, mode='markers', name='AL Sinyali', marker=dict(symbol='triangle-up', size=15, color='lime')))
        sells = df[df['Cross'] == -1]
        fig.add_trace(go.Scatter(x=sells.index, y=sells['SMA_SHORT']*1.01, mode='markers', name='SAT Sinyali', marker=dict(symbol='triangle-down', size=15, color='red')))

        fig.update_layout(template="plotly_dark", height=550, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 7. KARAR TABLOSU ---
        last = df.iloc[-1]
        res = []
        
        # Sinyaller
        res.append(["AL" if last['SMA_SHORT'] > last['SMA_LONG'] else "SAT", "SMA Crossover", "Trend yönü tespiti."])
        res.append(["AL" if last['RSI'] < 30 else "SAT" if last['RSI'] > 70 else "TUT", "RSI (14)", f"Seviye: {last['RSI']:.1f}"])
        res.append(["AL" if last['Close'].item() < last['Low'].item() else "SAT" if last['Close'].item() > last['Up'].item() else "TUT", "Bollinger Bands", "Fiyatın kanaldaki yeri."])
        res.append(["AL" if last['MACD'] > last['MACD_S'] else "SAT", "MACD", "Momentum durumu."])
        res.append(["AL" if last['Z'] < -2 else "SAT" if last['Z'] > 2 else "TUT", "Mean Reversion", "Ortalamadan sapma."])

        al_count = len([x for x in res if x[0] == "AL"])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Anlık Fiyat", f"{last['Close'].item():.2f}")
        c2.metric("Güven Skoru", f"{al_count}/5 AL")
        c3.metric("Zaman Dilimi", f"{interval}")

        st.subheader("🔍 Algoritmik Detaylar")
        res_df = pd.DataFrame(res, columns=["Karar", "Algoritma", "Durum/Sebep"])
        
        def color_map(val):
            color = '#00ff00' if val == 'AL' else '#ff4b4b' if val == 'SAT' else '#808495'
            return f'color: {color}; font-weight: bold'
        
        st.table(res_df.style.applymap(color_map, subset=['Karar']))

    else:
        st.error("Veri çekilemedi. Lütfen Ticker'ın doğruluğunu veya İnternet bağlantınızı kontrol edin.")
