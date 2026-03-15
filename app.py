import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ============================================================
# 1. SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(page_title="Algo-Trader Pro v3.1", layout="wide")

# 2. OTOMATİK YENİLEME (55 sn → cache TTL=60 sn ile çakışma önlenir)
st_autorefresh(interval=55 * 1000, key="terminal_refresh")

st.title("📈 Yatırım Algoritmaları Terminali")
st.caption("Piyasa verileri Yahoo Finance üzerinden 1 dakika gecikmeli/canlı olarak çekilmektedir.")

# ============================================================
# 3. YAN PANEL (SIDEBAR) - KONTROL MERKEZİ
# ============================================================
with st.sidebar:
    st.header("⚙️ Veri & Algoritma Ayarları")
    ticker = st.text_input("Ticker Sembolü:", "GC=F")

    # PERİYOT SEÇİMİ
    period = st.selectbox(
        "Toplam Veri Süresi (Period):",
        options=["1d", "5d", "1mo", "6mo", "1y", "5y", "max"],
        index=1,
    )

    # ARALIK (INTERVAL) SEÇİMİ
    if period in ["1d", "5d"]:
        interval_options = ["1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"]
        default_int_idx = 0
    elif period == "1mo":
        interval_options = ["2m", "5m", "15m", "30m", "60m", "1h", "1d"]
        default_int_idx = 6
    else:
        interval_options = ["1h", "1d", "1wk", "1mo"]
        default_int_idx = 1

    interval = st.selectbox(
        "Mum Aralığı (Interval):", options=interval_options, index=default_int_idx
    )

    st.write("---")
    st.subheader("Algoritma Hassasiyeti")
    sma_s = st.slider("Hızlı SMA (Kısa):", 5, 50, 20)
    sma_l = st.slider("Yavaş SMA (Uzun):", 50, 200, 100)

    st.write("---")
    st.subheader("RSI & Z-Score Eşikleri")
    rsi_lower = st.slider("RSI Alt Eşik (Aşırı Satım):", 10, 40, 30)
    rsi_upper = st.slider("RSI Üst Eşik (Aşırı Alım):", 60, 90, 70)
    z_threshold = st.slider("Z-Score Eşik (±):", 1.0, 3.0, 2.0, step=0.1)

    st.write("---")
    st.info(
        "İpucu: 1 dakikalık analizler için Periyot: 5d, Mum Aralığı: 1m seçiniz."
    )


# ============================================================
# 4. YARDIMCI FONKSİYONLAR
# ============================================================
def safe_scalar(value):
    """Pandas Series / numpy scalar'ı güvenli şekilde Python float'a çevirir."""
    if isinstance(value, (pd.Series, np.ndarray)):
        return float(value.iloc[0]) if len(value) > 0 else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """MultiIndex sütunları güvenli şekilde düzleştirir."""
    if isinstance(df.columns, pd.MultiIndex):
        # Tek ticker varsa sadece ilk seviyeyi al
        unique_tickers = df.columns.get_level_values(1).unique()
        if len(unique_tickers) <= 1:
            df.columns = df.columns.get_level_values(0)
        else:
            # Birden fazla ticker varsa (ticker, metric) formatına çevir
            df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    return df


# ============================================================
# 5. VERİ ÇEKME MOTORU
# ============================================================
@st.cache_data(ttl=50)
def fetch_live_data(symbol: str, p: str, i: str) -> pd.DataFrame:
    try:
        data = yf.download(symbol, period=p, interval=i, progress=False)
        if data is None or data.empty:
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()


# ============================================================
# 6. ANA MANTIK
# ============================================================
if ticker:
    df = fetch_live_data(ticker, period, interval)

    if not df.empty:
        df = flatten_columns(df)

        # Gerekli sütunların varlığını kontrol et
        required_cols = ["Open", "High", "Low", "Close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Eksik sütunlar: {missing}. Ticker veya veri kaynağını kontrol edin.")
            st.stop()

        close = df["Close"].squeeze()

        # Minimum veri kontrolü
        min_required = max(sma_l, 30)  # En az uzun SMA veya Z-Score penceresi kadar veri lazım
        if len(close) < min_required:
            st.warning(
                f"Yeterli veri yok: {len(close)} mum var, en az {min_required} gerekli. "
                f"Periyodu artırın veya interval'ı küçültün."
            )

        # -------------------------------------------------------
        # ALGORİTMA HESAPLAMALARI
        # -------------------------------------------------------

        # SMA Crossover
        df["SMA_SHORT"] = close.rolling(window=sma_s, min_periods=sma_s).mean()
        df["SMA_LONG"] = close.rolling(window=sma_l, min_periods=sma_l).mean()
        df["Sig"] = 0
        df.loc[df["SMA_SHORT"] > df["SMA_LONG"], "Sig"] = 1
        df["Cross"] = df["Sig"].diff()

        # RSI (division by zero korumalı)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)  # sıfıra bölme koruması
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["Mid"] = close.rolling(window=20).mean()
        df["Std"] = close.rolling(window=20).std()
        df["Up"] = df["Mid"] + (df["Std"] * 2)
        df["Low_BB"] = df["Mid"] - (df["Std"] * 2)  # 'Low' OHLC sütunuyla çakışmasın

        # MACD
        e12 = close.ewm(span=12, adjust=False).mean()
        e26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = e12 - e26
        df["MACD_S"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Z-Score
        z_mean = close.rolling(30).mean()
        z_std = close.rolling(30).std().replace(0, np.nan)
        df["Z"] = (close - z_mean) / z_std

        # -------------------------------------------------------
        # GRAFİK
        # -------------------------------------------------------
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Fiyat",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_SHORT"],
                name=f"SMA {sma_s}",
                line=dict(color="orange"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_LONG"],
                name=f"SMA {sma_l}",
                line=dict(color="cyan"),
            )
        )

        # AL/SAT okları → fiyata yakın konumlandırma
        buys = df[df["Cross"] == 1]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["Low"] * 0.995,
                    mode="markers",
                    name="AL Sinyali",
                    marker=dict(symbol="triangle-up", size=15, color="lime"),
                )
            )

        sells = df[df["Cross"] == -1]
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells["High"] * 1.005,
                    mode="markers",
                    name="SAT Sinyali",
                    marker=dict(symbol="triangle-down", size=15, color="red"),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------
        # KARAR TABLOSU
        # -------------------------------------------------------
        last = df.iloc[-1]

        last_close = safe_scalar(last["Close"])
        last_sma_s = safe_scalar(last["SMA_SHORT"])
        last_sma_l = safe_scalar(last["SMA_LONG"])
        last_rsi = safe_scalar(last["RSI"])
        last_up = safe_scalar(last["Up"])
        last_low_bb = safe_scalar(last["Low_BB"])
        last_macd = safe_scalar(last["MACD"])
        last_macd_s = safe_scalar(last["MACD_S"])
        last_z = safe_scalar(last["Z"])

        res = []

        # 1 - SMA Crossover
        if not np.isnan(last_sma_s) and not np.isnan(last_sma_l):
            res.append([
                "AL" if last_sma_s > last_sma_l else "SAT",
                "SMA Crossover",
                "Trend yönü tespiti.",
            ])
        else:
            res.append(["N/A", "SMA Crossover", "Yetersiz veri."])

        # 2 - RSI
        if not np.isnan(last_rsi):
            if last_rsi < rsi_lower:
                rsi_decision = "AL"
            elif last_rsi > rsi_upper:
                rsi_decision = "SAT"
            else:
                rsi_decision = "TUT"
            res.append([rsi_decision, "RSI (14)", f"Seviye: {last_rsi:.1f}"])
        else:
            res.append(["N/A", "RSI (14)", "Yetersiz veri."])

        # 3 - Bollinger Bands
        if not any(np.isnan(v) for v in [last_close, last_low_bb, last_up]):
            if last_close < last_low_bb:
                bb_decision = "AL"
            elif last_close > last_up:
                bb_decision = "SAT"
            else:
                bb_decision = "TUT"
            res.append([bb_decision, "Bollinger Bands", "Fiyatın kanaldaki yeri."])
        else:
            res.append(["N/A", "Bollinger Bands", "Yetersiz veri."])

        # 4 - MACD
        if not np.isnan(last_macd) and not np.isnan(last_macd_s):
            res.append([
                "AL" if last_macd > last_macd_s else "SAT",
                "MACD",
                "Momentum durumu.",
            ])
        else:
            res.append(["N/A", "MACD", "Yetersiz veri."])

        # 5 - Mean Reversion (Z-Score)
        if not np.isnan(last_z):
            if last_z < -z_threshold:
                z_decision = "AL"
            elif last_z > z_threshold:
                z_decision = "SAT"
            else:
                z_decision = "TUT"
            res.append([z_decision, "Mean Reversion", f"Z: {last_z:.2f}"])
        else:
            res.append(["N/A", "Mean Reversion", "Yetersiz veri."])

        # Güven Skoru (N/A olanları hariç tut)
        valid_signals = [x for x in res if x[0] != "N/A"]
        al_count = len([x for x in valid_signals if x[0] == "AL"])
        total_valid = len(valid_signals)

        c1, c2, c3 = st.columns(3)
        c1.metric("Anlık Fiyat", f"{last_close:.2f}")
        c2.metric("Güven Skoru", f"{al_count}/{total_valid} AL")
        c3.metric("Zaman Dilimi", f"{interval}")

        st.subheader("🔍 Algoritmik Detaylar")
        res_df = pd.DataFrame(res, columns=["Karar", "Algoritma", "Durum/Sebep"])

        def color_map(val):
            if val == "AL":
                return "color: #00ff00; font-weight: bold"
            elif val == "SAT":
                return "color: #ff4b4b; font-weight: bold"
            elif val == "N/A":
                return "color: #ffaa00; font-weight: bold"
            return "color: #808495; font-weight: bold"

        st.table(res_df.style.map(color_map, subset=["Karar"]))

    else:
        st.error(
            "Veri çekilemedi. Lütfen Ticker'ın doğruluğunu veya "
            "İnternet bağlantınızı kontrol edin."
        )
