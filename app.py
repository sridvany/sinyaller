import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Algo-Trader Pro v2.0", layout="wide")

# 2. OTOMATİK YENİLEME (Her 60 saniyede bir sayfayı tazeler)
st_autorefresh(interval=60 * 1000, key="data_refresh")

st.title("📈 Yatırım Algoritmaları Simülatörü")
st.caption("Veriler 60 saniyede bir otomatik olarak güncellenir.")

# 3. YAN PANEL (SIDEBAR) AYARLARI
with st.sidebar:
    st.header("⚙️ Terminal Ayarları")
    ticker = st.text_input("Ticker Giriniz:", "GC=F")
    
    period = st.selectbox(
        "Zaman Aralığı (Period):",
        options=["1d", "5d", "1mo", "6mo", "1y", "5y", "max"],
        index=4  # Varsayılan: 1y
    )
    
    # Periyoda göre uygun intervalleri filtrele
    if period in ["1d", "5d"]:
        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m"]
        default_idx = 2 # 5m
    else:
        interval_options = ["1h", "1d", "1wk", "1mo"]
        default_idx = 1 # 1d
        
    interval = st.selectbox("Mum Aralığı (Interval):", options=interval_options, index=default_idx)

    st.write("---")
    st.subheader("Algoritma Hassasiyeti")
    sma_short_val = st.slider("Kısa SMA:", 5, 50, 20)
    sma_long_val = st.slider("Uzun SMA:", 50, 200, 50)
    
    st.write("---")
    st.warning("Bu uygulama yatırım tavsiyesi içermez.")

# 4. VERİ ÇEKME FONKSİYONU
@st.cache_data(ttl=60) 
def get_data(symbol, p, i):
    try:
        df = yf.download(symbol, period=p, interval=i)
        return df
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()

if ticker:
    df = get_data(ticker, period, interval)
    
    if not df.empty:
        # MultiIndex Sütun Düzeltmesi (Hata engelleyici)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Veriyi seriye dönüştür (Hesaplamalar için)
        close = df['Close'].squeeze()
        
        # --- 5. TEKNİK ANALİZ HESAPLAMALARI ---
        
        # SMA & Sinyaller
        df['SMA_S'] = close.rolling(window=sma_short_val).mean()
        df['SMA_L'] = close.rolling(window=sma_long_val).mean()
        df['Signal'] = 0
        df.loc[df['SMA_S'] > df['SMA_L'], 'Signal'] = 1
        df['Crossover'] = df['Signal'].diff()

        # RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Mid'] = close.rolling(window=20).mean()
        df['BB_Std'] = close.rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Z-Score (Mean Reversion)
        df['Mean'] = close.rolling(window=30).mean()
        df['Std'] = close.rolling(window=30).std()
        df['Z_Score'] = (close - df['Mean']) / df['Std']

        # --- 6. PROFESYONEL GRAFİK (PLOTLY) ---
        fig = go.Figure()
        
        # Mum Grafik
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], 
            low=df['Low'], close=df['Close'], name='Fiyat'
        ))
        
        # SMA Çizgileri
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_S'], name=f'SMA {sma_short_val}', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_L'], name=f'SMA {sma_long_val}', line=dict(color='blue', width=1.5)))

        # Al/Sat İşaretçileri
        buy_points = df[df['Crossover'] == 1]
        fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['SMA_S']*0.98, mode='markers', name='AL (Kesişme)', marker=dict(symbol='triangle-up', size=15, color='#00ff00')))
        
        sell_points = df[df['Crossover'] == -1]
        fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points['SMA_S']*1.02, mode='markers', name='SAT (Kesişme)', marker=dict(symbol='triangle-down', size=15, color='#ff4b4b')))

        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # --- 7. KARAR MEKANİZMASI VE TABLO ---
        last_row = df.iloc[-1]
        results = []

        # Karar Mantığı
        # 1. SMA
        if last_row['SMA_S'] > last_row['SMA_L']:
            results.append(["AL", "SMA Crossover", "Yükseliş trendi teyit edildi."])
        else:
            results.append(["SAT", "SMA Crossover", "Düşüş baskısı hakim."])

        # 2. RSI
        if last_row['RSI'] < 30:
            results.append(["AL", "RSI (14)", f"Aşırı satış ({last_row['RSI']:.1f}). Tepki alımı yakın."])
        elif last_row['RSI'] > 70:
            results.append(["SAT", "RSI (14)", f"Aşırı alım ({last_row['RSI']:.1f}). Düzeltme beklenebilir."])
        else:
            results.append(["TUT", "RSI (14)", f"Nötr ({last_row['RSI']:.1f}). Momentum dengeli."])

        # 3. Bollinger
        if last_row['Close'].item() < last_row['BB_Lower'].item():
            results.append(["AL", "Bollinger Bands", "Fiyat alt banttan sekti. İstatistiki fırsat."])
        elif last_row['Close'].item() > last_row['BB_Upper'].item():
            results.append(["SAT", "Bollinger Bands", "Fiyat üst bandı zorluyor. Geri çekilme riski."])
        else:
            results.append(["TUT", "Bollinger Bands", "Fiyat kanal içinde."])

        # 4. MACD
        if last_row['MACD'] > last_row['MACD_Signal']:
            results.append(["AL", "MACD", "Pozitif ivme artışı."])
        else:
            results.append(["SAT", "MACD", "Negatif ivme artışı."])

        # 5. Z-Score
        if last_row['Z_Score'] < -2:
            results.append(["AL", "Mean Reversion", "Fiyat ortalamadan çok saptı (Ucuz)."])
        elif last_row['Z_Score'] > 2:
            results.append(["SAT", "Mean Reversion", "Fiyat ortalamadan çok saptı (Pahalı)."])
        else:
            results.append(["TUT", "Mean Reversion", "Fiyat ortalamaya yakın."])

        # ÖZET SKOR KARTI
        al_sayisi = len([r for r in results if r[0] == "AL"])
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Anlık Fiyat", f"{last_row['Close'].item():.2f}")
        with c2: st.metric("Algoritmik Güven Skoru", f"{al_sayisi}/5 AL")
        with c3: st.metric("İşlem Hacmi", f"{int(last_row['Volume'].item()):,}")

        # TABLO GÖRÜNÜMÜ
        res_df = pd.DataFrame(results, columns=["Karar", "Algoritma", "Sebep"])
        def color_decision(val):
            color = '#00ff00' if val == 'AL' else '#ff4b4b' if val == 'SAT' else '#808495'
            return f'color: {color}; font-weight: bold'
        
        st.subheader("📊 Algoritmik Sinyal Analizi")
        st.table(res_df.style.applymap(color_decision, subset=['Karar']))

    else:
        st.warning("Veri bulunamadı. Lütfen Ticker'ı kontrol edin.")
