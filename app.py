import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Sayfa Konfigürasyonu
st.set_page_config(page_title="Algo-Trader Signal Pro", layout="wide")

st.title("📈 Yatırım Simülatörü")
ticker = st.text_input("Hisse veya Emtia için ilgili varlığın ticker'ını Gemini'ye sorabilirsiniz. Bu bir simülatördür yatırım tavsiyesi değildir. (Örn: AAPL, GC=F, BTC-USD):", "GC=F")

# Veri Çekme
@st.cache_data
def get_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    return df

if ticker:
    data = get_data(ticker)
    
    if not data.empty:
        # --- ALGORİTMA HESAPLAMALARI ---
        close = data['Close']
        
        # 1. SMA Crossover (Trend Takibi)
        sma50 = close.rolling(window=50).mean()
        sma200 = close.rolling(window=200).mean()
        
        # 2. RSI (Momentum)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 3. Bollinger Bands (Volatilite)
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)
        
        # 4. MACD (Trend Momentum)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # 5. Mean Reversion (Z-Score)
        mean = close.rolling(window=30).mean()
        std = close.rolling(window=30).std()
        z_score = (close - mean) / std

        # --- KARAR MEKANİZMASI ---
        last_close = close.iloc[-1].item()
        results = []

        # Karar Logic
        if sma50.iloc[-1].item() > sma200.iloc[-1].item():
            results.append(["AL", "SMA Crossover", "Altın Kesişme (50 günlük ortalama 200'ü yukarı kesti). Uzun vadeli yükseliş trendi."])
        else:
            results.append(["SAT", "SMA Crossover", "Ölüm Kesişmesi (50 günlük ortalama 200'ün altında). Ayı piyasası riski."])

        if rsi.iloc[-1].item() < 30:
            results.append(["AL", "RSI (Momentum)", f"RSI değeri {rsi.iloc[-1].item():.2f} ile aşırı satış bölgesinde. Tepki alımı gelebilir."])
        elif rsi.iloc[-1].item() > 70:
            results.append(["SAT", "RSI (Momentum)", f"RSI değeri {rsi.iloc[-1].item():.2f} ile aşırı alım bölgesinde. Düzeltme beklenebilir."])
        else:
            results.append(["TUT", "RSI (Momentum)", "RSI nötr bölgede (30-70 arası)."])

        if last_close < bb_lower.iloc[-1].item():
            results.append(["AL", "Bollinger Bands", "Fiyat alt banttan dışarı taştı. İstatistiksel olarak yukarı dönme ihtimali yüksek."])
        elif last_close > bb_upper.iloc[-1].item():
            results.append(["SAT", "Bollinger Bands", "Fiyat üst banta çarptı. Aşırı genişleme var."])

        if macd.iloc[-1].item() > signal_line.iloc[-1].item():
            results.append(["AL", "MACD", "MACD çizgisi sinyal çizgisini yukarı kesti. Pozitif ivme artıyor."])
        else:
            results.append(["SAT", "MACD", "MACD sinyal çizgisinin altında. Negatif ivme hakim."])

        if z_score.iloc[-1].item() < -2:
            results.append(["AL", "Mean Reversion", "Fiyat ortalamanın 2 standart sapma altında. Ortalamaya dönüş (Reversion) beklenir."])

        # Arayüz Çıktısı
        st.subheader(f"{ticker} İçin Algoritmik Sinyal Raporu")
        res_df = pd.DataFrame(results, columns=["Karar", "Algoritma", "Sebep"])
        
        def color_decision(val):
            color = 'green' if val == 'AL' else 'red' if val == 'SAT' else 'gray'
            return f'color: {color}; font-weight: bold'

        st.table(res_df.style.applymap(color_decision, subset=['Karar']))
        
        # Grafik
        st.line_chart(data['Close'])
