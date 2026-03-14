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
        # --- KRİTİK DÜZELTME BURADA ---
        # Eğer yfinance veriyi katmanlı (MultiIndex) gönderirse, onu düzleştiriyoruz.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # --- 1. TEKNİK HESAPLAMALAR ---
        close = df['Close']
        
        # SMA & Sinyaller
        df['SMA50'] = close.rolling(window=50).mean()
        df['SMA200'] = close.rolling(window=200).mean()
        df['Signal'] = 0
        df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = 1
        df['Crossover'] = df['Signal'].diff()

        # RSI
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

        # --- 2. PROFESYONEL GRAFİK ---
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50', line=dict(color='orange', width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA 200', line=dict(color='blue', width=1.5)))

        # Al/Sat Okları
        buy_signals = df[df['Crossover'] == 1]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['SMA50']*0.97, mode='markers', name='AL (Kesişme)', marker=dict(symbol='triangle-up', size=12, color='green')))
        sell_signals = df[df['Crossover'] == -1]
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['SMA50']*1.03, mode='markers', name='SAT (Kesişme)', marker=dict(symbol='triangle-down', size=12, color='red')))

        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- 3. ALGORİTMİK KARAR TABLOSU ---
        st.subheader("📊 Algoritmik Sinyal Raporu")
        
        last_row = df.iloc[-1]
        results = []

        # SMA Kararı
        if last_row['SMA50'] > last_row['SMA200']:
            results.append(["AL", "SMA Crossover", "Altın Kesişme: Yükseliş trendi devam ediyor."])
        else:
            results.append(["SAT", "SMA Crossover", "Ölüm Kesişmesi: Ayı piyasası baskın."])

        # RSI Kararı
        if last_row['RSI'] < 35:
            results.append(["AL", "RSI (14)", f"RSI: {last_row['RSI']:.2f}. Aşırı satış, tepki gelebilir."])
        elif last_row['RSI'] > 65:
            results.append(["SAT", "RSI (14)", f"RSI: {last_row['RSI']:.2f}. Aşırı alım, düzeltme riski."])
        else:
            results.append(["TUT", "RSI (14)", "Nötr bölge, momentum dengeli."])

        # Bollinger Kararı
        if last_row['Close'].item() < last_row['BB_Lower'].item():
            results.append(["AL", "Bollinger Bands", "Fiyat alt bandın dışında. İstatistiksel geri dönüş beklentisi."])
        elif last_row['Close'].item() > last_row['BB_Upper'].item():
            results.append(["SAT", "Bollinger Bands", "Fiyat üst bandın dışında. Aşırı genişleme."])
        else:
            results.append(["TUT", "Bollinger Bands", "Fiyat bant aralığında seyrediyor."])

        # MACD Kararı
        if last_row['MACD'] > last_row['MACD_Signal']:
            results.append(["AL", "MACD", "MACD çizgisi sinyalin üzerinde. Pozitif momentum."])
        else:
            results.append(["SAT", "MACD", "MACD çizgisi sinyalin altında. Negatif ivme."])

        # Mean Reversion Kararı
        if last_row['Z_Score'] < -2:
            results.append(["AL", "Mean Reversion", "Z-Score çok düşük. Ortalamaya dönüş hareketi başlayabilir."])
        elif last_row['Z_Score'] > 2:
            results.append(["SAT", "Mean Reversion", "Z-Score çok yüksek. Ortalamaya doğru geri çekilme muhtemel."])
        else:
            results.append(["TUT", "Mean Reversion", "Fiyat ortalama değerine yakın."])

        # Tabloyu Renklendirerek Göster
        res_df = pd.DataFrame(results, columns=["Karar", "Algoritma", "Sebep"])
        
        def color_decision(val):
            color = '#00ff00' if val == 'AL' else '#ff4b4b' if val == 'SAT' else '#808495'
            return f'color: {color}; font-weight: bold'

        st.table(res_df.style.applymap(color_decision, subset=['Karar']))

# Tablonun hemen altına eklenebilir
al_sayisi = len([r for r in results if r[0] == "AL"])
st.metric(label="Algoritmik Güven Skoru", value=f"{al_sayisi}/5 AL", delta=f"{al_sayisi*20}% Pozitif")
