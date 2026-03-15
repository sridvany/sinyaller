import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ============================================================
# 1. SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(page_title="Algo-Trader Pro v3.3", layout="wide")

# 2. OTOMATİK YENİLEME (55 sn → cache TTL=50 sn ile uyumlu)
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
    st.subheader("ADX & ATR Ayarları")
    adx_period = st.slider("ADX Periyodu:", 7, 30, 14)
    adx_threshold = st.slider("ADX Trend Eşiği:", 15, 40, 25)
    atr_period = st.slider("ATR Periyodu:", 7, 30, 14)

    st.write("---")
    st.subheader("Stochastic RSI Ayarları")
    stoch_rsi_period = st.slider("Stoch RSI Periyodu:", 7, 21, 14)
    stoch_lower = st.slider("Stoch RSI Alt Eşik:", 5, 30, 20)
    stoch_upper = st.slider("Stoch RSI Üst Eşik:", 70, 95, 80)

    st.write("---")
    st.subheader("Ichimoku Ayarları")
    ichi_tenkan = st.slider("Tenkan-sen (Dönüşüm):", 5, 20, 9)
    ichi_kijun = st.slider("Kijun-sen (Baz):", 20, 40, 26)
    ichi_senkou_b = st.slider("Senkou Span B:", 40, 65, 52)

    st.write("---")
    st.subheader("Konsensüs Ayarları")
    consensus_threshold = st.slider(
        "Minimum AL/SAT çoğunluğu (10 sinyalden):", 3, 9, 6
    )

    st.write("---")
    st.subheader("📊 Backtest Ayarları")
    commission_pct = st.slider("Komisyon (% / işlem):", 0.0, 1.0, 0.1, step=0.01)
    slippage_pct = st.slider("Slippage (% / işlem):", 0.0, 0.5, 0.05, step=0.01)
    initial_capital = st.number_input("Başlangıç Sermayesi ($):", min_value=100, value=10000, step=100)

    st.write("---")
    st.subheader("🧬 Parametre Optimizasyonu")
    run_optimization = st.button("🚀 Optimize Et", use_container_width=True)
    st.caption("Walk-forward: %70 in-sample, %30 out-of-sample")

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
        unique_tickers = df.columns.get_level_values(1).unique()
        if len(unique_tickers) <= 1:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    return df


def calc_adx(high, low, close, period=14):
    """
    ADX (Average Directional Index) hesaplama.
    Wilder smoothing kullanır.
    Döndürür: adx, plus_di, minus_di Series'leri.
    """
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=high.index, dtype=float)

    # Wilder Smoothing (EMA with alpha=1/period)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # DI hesaplama
    plus_di = 100 * (smooth_plus / atr.replace(0, np.nan))
    minus_di = 100 * (smooth_minus / atr.replace(0, np.nan))

    # DX ve ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx, plus_di, minus_di


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
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Eksik sütunlar: {missing}. Ticker veya veri kaynağını kontrol edin.")
            st.stop()

        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        # Minimum veri kontrolü
        min_required = max(sma_l, 30, adx_period * 3)
        if len(close) < min_required:
            st.warning(
                f"Yeterli veri yok: {len(close)} mum var, en az {min_required} gerekli. "
                f"Periyodu artırın veya interval'ı küçültün."
            )

        # -------------------------------------------------------
        # ALGORİTMA HESAPLAMALARI
        # -------------------------------------------------------

        # 1. SMA Crossover
        df["SMA_SHORT"] = close.rolling(window=sma_s, min_periods=sma_s).mean()
        df["SMA_LONG"] = close.rolling(window=sma_l, min_periods=sma_l).mean()
        df["Sig_SMA"] = np.where(df["SMA_SHORT"] > df["SMA_LONG"], 1, -1)
        df.loc[df["SMA_SHORT"].isna() | df["SMA_LONG"].isna(), "Sig_SMA"] = 0

        # 2. RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        df["Sig_RSI"] = np.where(df["RSI"] < rsi_lower, 1, np.where(df["RSI"] > rsi_upper, -1, 0))

        # 3. Bollinger Bands
        df["Mid"] = close.rolling(window=20).mean()
        df["Std"] = close.rolling(window=20).std()
        df["Up"] = df["Mid"] + (df["Std"] * 2)
        df["Low_BB"] = df["Mid"] - (df["Std"] * 2)
        df["Sig_BB"] = np.where(close < df["Low_BB"], 1, np.where(close > df["Up"], -1, 0))

        # 4. MACD
        e12 = close.ewm(span=12, adjust=False).mean()
        e26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = e12 - e26
        df["MACD_S"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["Sig_MACD"] = np.where(df["MACD"] > df["MACD_S"], 1, -1)

        # 5. Mean Reversion (Z-Score)
        z_mean = close.rolling(30).mean()
        z_std = close.rolling(30).std().replace(0, np.nan)
        df["Z"] = (close - z_mean) / z_std
        df["Sig_Z"] = np.where(df["Z"] < -z_threshold, 1, np.where(df["Z"] > z_threshold, -1, 0))

        # 6. OBV (On-Balance Volume)
        obv_sign = np.sign(close.diff()).fillna(0)
        df["OBV"] = (volume * obv_sign).cumsum()
        obv_sma_short = df["OBV"].rolling(window=10, min_periods=10).mean()
        obv_sma_long = df["OBV"].rolling(window=30, min_periods=30).mean()
        df["Sig_OBV"] = np.where(obv_sma_short > obv_sma_long, 1, -1)
        df.loc[obv_sma_short.isna() | obv_sma_long.isna(), "Sig_OBV"] = 0

        # 7. ATR (Average True Range) — volatilite filtresi
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
        atr_ma = df["ATR"].rolling(window=30, min_periods=30).mean()
        # ATR filtresi: düşük volatilitede TUT (0), yüksekte mevcut trende eşlik et
        df["ATR_High"] = df["ATR"] > atr_ma
        # ATR tek başına AL/SAT vermez, konsensüste filtre olarak kullanılacak

        # 8. ADX (Average Directional Index)
        df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = calc_adx(high, low, close, period=adx_period)
        # ADX > eşik → trend var, DI yönüne göre sinyal
        # ADX < eşik → trend yok, mean reversion sinyallerini güçlendir
        df["Sig_ADX"] = np.where(
            df["ADX"] > adx_threshold,
            np.where(df["PLUS_DI"] > df["MINUS_DI"], 1, -1),  # Trendli: DI yönü
            0  # Trendsiz: nötr
        )

        # 9. VWAP (sadece intraday interval'larda aktif)
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"]
        if is_intraday and "Volume" in df.columns:
            typical_price = (high + low + close) / 3
            cum_vol = volume.cumsum()
            cum_tp_vol = (typical_price * volume).cumsum()
            df["VWAP"] = cum_tp_vol / cum_vol.replace(0, np.nan)
            df["Sig_VWAP"] = np.where(close > df["VWAP"], 1, -1)
            df.loc[df["VWAP"].isna(), "Sig_VWAP"] = 0
        else:
            df["VWAP"] = np.nan
            df["Sig_VWAP"] = 0  # Günlük+ → devre dışı, konsensüse katılmaz

        # 10. Stochastic RSI
        rsi_series = df["RSI"]
        rsi_min = rsi_series.rolling(window=stoch_rsi_period, min_periods=stoch_rsi_period).min()
        rsi_max = rsi_series.rolling(window=stoch_rsi_period, min_periods=stoch_rsi_period).max()
        rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
        df["StochRSI_K"] = ((rsi_series - rsi_min) / rsi_range) * 100
        df["StochRSI_D"] = df["StochRSI_K"].rolling(window=3).mean()  # %D = 3-period SMA of %K
        df["Sig_StochRSI"] = np.where(
            df["StochRSI_K"] < stoch_lower, 1,
            np.where(df["StochRSI_K"] > stoch_upper, -1, 0)
        )

        # 11. Ichimoku Cloud
        tenkan = (high.rolling(window=ichi_tenkan).max() + low.rolling(window=ichi_tenkan).min()) / 2
        kijun = (high.rolling(window=ichi_kijun).max() + low.rolling(window=ichi_kijun).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(ichi_kijun)
        senkou_b = ((high.rolling(window=ichi_senkou_b).max() + low.rolling(window=ichi_senkou_b).min()) / 2).shift(ichi_kijun)
        df["Tenkan"] = tenkan
        df["Kijun"] = kijun
        df["Senkou_A"] = senkou_a
        df["Senkou_B"] = senkou_b
        df["Chikou"] = close.shift(-ichi_kijun)

        # Ichimoku sinyal: Tenkan > Kijun + fiyat bulutun üstünde = AL, tersi SAT
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        ichi_bullish = (tenkan > kijun) & (close > cloud_top)
        ichi_bearish = (tenkan < kijun) & (close < cloud_bottom)
        df["Sig_Ichimoku"] = np.where(ichi_bullish, 1, np.where(ichi_bearish, -1, 0))

        # -------------------------------------------------------
        # KONSENSÜS HESABI (her satır için)
        # -------------------------------------------------------
        signal_cols = ["Sig_SMA", "Sig_RSI", "Sig_BB", "Sig_MACD", "Sig_Z",
                       "Sig_OBV", "Sig_ADX", "Sig_VWAP", "Sig_StochRSI", "Sig_Ichimoku"]

        # ATR filtresi: düşük volatilitede trend sinyallerini bastır
        trend_signals = ["Sig_SMA", "Sig_MACD"]
        for col in trend_signals:
            df[col] = np.where(df["ATR_High"] | (df[col] == 0), df[col], 0)

        # AL sayısı (sinyal=1) ve SAT sayısı (sinyal=-1)
        sig_df = df[signal_cols]
        df["AL_Count"] = (sig_df == 1).sum(axis=1)
        df["SAT_Count"] = (sig_df == -1).sum(axis=1)
        df["Valid_Count"] = (sig_df != 0).sum(axis=1)

        # Konsensüs: eşik aşılırsa sinyal, aksi halde nötr
        df["Consensus"] = 0
        df.loc[df["AL_Count"] >= consensus_threshold, "Consensus"] = 1
        df.loc[df["SAT_Count"] >= consensus_threshold, "Consensus"] = -1

        # Konsensüs değişim noktaları (ok basılacak yerler)
        df["Consensus_Change"] = df["Consensus"].diff()

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

        # VWAP (intraday'de görünür)
        if is_intraday:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["VWAP"],
                    name="VWAP",
                    line=dict(color="yellow", dash="dash", width=1.5),
                )
            )

        # Konsensüs bazlı AL/SAT okları
        buys = df[df["Consensus_Change"] > 0]  # Nötr/SAT → AL geçişi
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["Low"] * 0.995,
                    mode="markers",
                    name=f"AL ({consensus_threshold}+ sinyal)",
                    marker=dict(symbol="triangle-up", size=15, color="lime"),
                )
            )

        sells = df[df["Consensus_Change"] < 0]  # Nötr/AL → SAT geçişi
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells["High"] * 1.005,
                    mode="markers",
                    name=f"SAT ({consensus_threshold}+ sinyal)",
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
        # ALT GRAFİKLER: RSI, MACD, ADX, OBV
        # -------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["RSI", "MACD", "ADX", "OBV", "Stoch RSI", "Ichimoku"])

        with tab1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="magenta")))
            fig_rsi.add_hline(y=rsi_lower, line_dash="dash", line_color="lime", annotation_text=f"Aşırı Satım ({rsi_lower})")
            fig_rsi.add_hline(y=rsi_upper, line_dash="dash", line_color="red", annotation_text=f"Aşırı Alım ({rsi_upper})")
            fig_rsi.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=30))
            st.plotly_chart(fig_rsi, use_container_width=True)

        with tab2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="cyan")))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_S"], name="Sinyal", line=dict(color="orange")))
            hist = df["MACD"] - df["MACD_S"]
            colors = ["lime" if v >= 0 else "red" for v in hist]
            fig_macd.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color=colors, opacity=0.5))
            fig_macd.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=30))
            st.plotly_chart(fig_macd, use_container_width=True)

        with tab3:
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX", line=dict(color="yellow", width=2)))
            fig_adx.add_trace(go.Scatter(x=df.index, y=df["PLUS_DI"], name="+DI", line=dict(color="lime", dash="dot")))
            fig_adx.add_trace(go.Scatter(x=df.index, y=df["MINUS_DI"], name="-DI", line=dict(color="red", dash="dot")))
            fig_adx.add_hline(y=adx_threshold, line_dash="dash", line_color="white", annotation_text=f"Trend Eşiği ({adx_threshold})")
            fig_adx.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=30))
            st.plotly_chart(fig_adx, use_container_width=True)

        with tab4:
            fig_obv = go.Figure()
            fig_obv.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="dodgerblue")))
            fig_obv.add_trace(go.Scatter(x=df.index, y=obv_sma_short, name="OBV SMA 10", line=dict(color="orange", dash="dot")))
            fig_obv.add_trace(go.Scatter(x=df.index, y=obv_sma_long, name="OBV SMA 30", line=dict(color="cyan", dash="dot")))
            fig_obv.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=30))
            st.plotly_chart(fig_obv, use_container_width=True)

        with tab5:
            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=df.index, y=df["StochRSI_K"], name="%K", line=dict(color="magenta")))
            fig_stoch.add_trace(go.Scatter(x=df.index, y=df["StochRSI_D"], name="%D", line=dict(color="orange", dash="dot")))
            fig_stoch.add_hline(y=stoch_lower, line_dash="dash", line_color="lime", annotation_text=f"Aşırı Satım ({stoch_lower})")
            fig_stoch.add_hline(y=stoch_upper, line_dash="dash", line_color="red", annotation_text=f"Aşırı Alım ({stoch_upper})")
            fig_stoch.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=30))
            st.plotly_chart(fig_stoch, use_container_width=True)

        with tab6:
            fig_ichi = go.Figure()
            fig_ichi.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                               low=df["Low"], close=df["Close"], name="Fiyat"))
            fig_ichi.add_trace(go.Scatter(x=df.index, y=df["Tenkan"], name="Tenkan-sen",
                                          line=dict(color="cyan", width=1)))
            fig_ichi.add_trace(go.Scatter(x=df.index, y=df["Kijun"], name="Kijun-sen",
                                          line=dict(color="red", width=1)))
            fig_ichi.add_trace(go.Scatter(x=df.index, y=df["Senkou_A"], name="Senkou A",
                                          line=dict(color="lime", width=0.5, dash="dot")))
            fig_ichi.add_trace(go.Scatter(x=df.index, y=df["Senkou_B"], name="Senkou B",
                                          line=dict(color="red", width=0.5, dash="dot"),
                                          fill="tonexty", fillcolor="rgba(100,100,100,0.15)"))
            fig_ichi.update_layout(template="plotly_dark", height=350, xaxis_rangeslider_visible=False,
                                   margin=dict(t=30, b=30))
            st.plotly_chart(fig_ichi, use_container_width=True)

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
        last_adx = safe_scalar(last["ADX"])
        last_plus_di = safe_scalar(last["PLUS_DI"])
        last_minus_di = safe_scalar(last["MINUS_DI"])
        last_obv_sig = safe_scalar(last["Sig_OBV"])
        last_atr_high = bool(last["ATR_High"]) if not pd.isna(last["ATR_High"]) else False

        res = []

        # 1 - SMA Crossover
        if not np.isnan(last_sma_s) and not np.isnan(last_sma_l):
            sma_decision = "AL" if last_sma_s > last_sma_l else "SAT"
            if not last_atr_high:
                sma_decision = "TUT (düşük vol.)"
            res.append([sma_decision, "SMA Crossover", "Trend yönü tespiti."])
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
            macd_decision = "AL" if last_macd > last_macd_s else "SAT"
            if not last_atr_high:
                macd_decision = "TUT (düşük vol.)"
            res.append([macd_decision, "MACD", "Momentum durumu."])
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

        # 6 - OBV
        if last_obv_sig != 0 and not np.isnan(last_obv_sig):
            obv_decision = "AL" if last_obv_sig > 0 else "SAT"
            res.append([obv_decision, "OBV", "Hacim trendi."])
        else:
            res.append(["N/A", "OBV", "Yetersiz veri."])

        # 7 - ADX
        if not np.isnan(last_adx):
            if last_adx > adx_threshold:
                adx_decision = "AL" if last_plus_di > last_minus_di else "SAT"
                adx_note = f"ADX: {last_adx:.1f} (Güçlü trend)"
            else:
                adx_decision = "TUT"
                adx_note = f"ADX: {last_adx:.1f} (Zayıf trend)"
            res.append([adx_decision, "ADX", adx_note])
        else:
            res.append(["N/A", "ADX", "Yetersiz veri."])

        # 8 - VWAP (intraday only)
        if is_intraday:
            last_vwap = safe_scalar(last["VWAP"])
            if not np.isnan(last_vwap):
                vwap_decision = "AL" if last_close > last_vwap else "SAT"
                res.append([vwap_decision, "VWAP", f"VWAP: {last_vwap:.2f}"])
            else:
                res.append(["N/A", "VWAP", "Yetersiz veri."])
        else:
            res.append(["N/A", "VWAP", "Günlük+ periyotta devre dışı."])

        # 9 - Stochastic RSI
        last_stoch_k = safe_scalar(last["StochRSI_K"])
        if not np.isnan(last_stoch_k):
            if last_stoch_k < stoch_lower:
                stoch_decision = "AL"
            elif last_stoch_k > stoch_upper:
                stoch_decision = "SAT"
            else:
                stoch_decision = "TUT"
            res.append([stoch_decision, "Stoch RSI", f"%K: {last_stoch_k:.1f}"])
        else:
            res.append(["N/A", "Stoch RSI", "Yetersiz veri."])

        # 10 - Ichimoku
        last_ichi_sig = safe_scalar(last["Sig_Ichimoku"])
        if last_ichi_sig == 1:
            res.append(["AL", "Ichimoku", "Tenkan > Kijun, fiyat bulut üstünde."])
        elif last_ichi_sig == -1:
            res.append(["SAT", "Ichimoku", "Tenkan < Kijun, fiyat bulut altında."])
        else:
            res.append(["TUT", "Ichimoku", "Karışık sinyal / bulut içinde."])

        # 8 - ATR Volatilite Durumu (bilgi amaçlı, AL/SAT vermez)
        last_atr = safe_scalar(last["ATR"])
        if not np.isnan(last_atr):
            vol_status = "Yüksek ↑" if last_atr_high else "Düşük ↓"
            res.append(["BİLGİ", "ATR Filtre", f"Volatilite: {vol_status} (ATR: {last_atr:.2f})"])
        else:
            res.append(["N/A", "ATR Filtre", "Yetersiz veri."])

        # Güven Skoru
        valid_signals = [x for x in res if x[0] in ("AL", "SAT")]
        al_count = len([x for x in valid_signals if x[0] == "AL"])
        sat_count = len([x for x in valid_signals if x[0] == "SAT"])
        total_valid = len(valid_signals)

        # Konsensüs kararı
        if al_count >= consensus_threshold:
            consensus_label = "🟢 AL"
        elif sat_count >= consensus_threshold:
            consensus_label = "🔴 SAT"
        else:
            consensus_label = "🟡 KARARSIZ"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anlık Fiyat", f"{last_close:.2f}")
        c2.metric("Güven Skoru", f"{al_count} AL / {sat_count} SAT")
        c3.metric("Konsensüs", consensus_label)
        c4.metric("Zaman Dilimi", f"{interval}")

        st.subheader("🔍 Algoritmik Detaylar")
        res_df = pd.DataFrame(res, columns=["Karar", "Algoritma", "Durum/Sebep"])

        def color_map(val):
            if val == "AL":
                return "color: #00ff00; font-weight: bold"
            elif val == "SAT":
                return "color: #ff4b4b; font-weight: bold"
            elif val == "N/A":
                return "color: #ffaa00; font-weight: bold"
            elif val == "BİLGİ":
                return "color: #00bfff; font-weight: bold"
            elif "düşük vol." in str(val):
                return "color: #808495; font-style: italic"
            return "color: #808495; font-weight: bold"

        st.table(res_df.style.map(color_map, subset=["Karar"]))

        # -------------------------------------------------------
        # BACKTEST MOTORU (Konsensüs Bazlı)
        # -------------------------------------------------------
        st.write("---")
        st.header("📊 Konsensüs Backtest Sonuçları")
        st.caption(
            "⚠️ Geçmiş performans gelecekteki sonuçların garantisi değildir. "
            "Backtest overfitting riski içerir — parametreleri geçmişe optimize etmek "
            "canlıda aynı sonucu garanti etmez."
        )

        # Trade listesi oluştur: Consensus AL'da gir, SAT'ta çık
        trades = []
        in_position = False
        entry_price = 0.0
        entry_date = None
        entry_idx = 0

        close_arr = close.values
        consensus_arr = df["Consensus"].values
        index_arr = df.index

        for i in range(1, len(df)):
            if not in_position and consensus_arr[i] == 1 and consensus_arr[i - 1] != 1:
                # AL sinyali — pozisyon aç
                entry_price = float(close_arr[i])
                entry_date = index_arr[i]
                entry_idx = i
                in_position = True
            elif in_position and consensus_arr[i] == -1 and consensus_arr[i - 1] != -1:
                # SAT sinyali — pozisyon kapat
                exit_price = float(close_arr[i])
                exit_date = index_arr[i]

                # Komisyon + slippage hesabı
                cost_pct = (commission_pct + slippage_pct) / 100
                net_entry = entry_price * (1 + cost_pct)
                net_exit = exit_price * (1 - cost_pct)
                pnl_pct = ((net_exit - net_entry) / net_entry) * 100
                duration = i - entry_idx

                trades.append({
                    "Giriş Tarihi": entry_date,
                    "Çıkış Tarihi": exit_date,
                    "Giriş Fiyatı": round(entry_price, 2),
                    "Çıkış Fiyatı": round(exit_price, 2),
                    "Getiri (%)": round(pnl_pct, 2),
                    "Süre (mum)": duration,
                })
                in_position = False

        # Açık pozisyon varsa son fiyatla kapat (bilgi amaçlı)
        if in_position:
            exit_price = float(close_arr[-1])
            cost_pct = (commission_pct + slippage_pct) / 100
            net_entry = entry_price * (1 + cost_pct)
            net_exit = exit_price * (1 - cost_pct)
            pnl_pct = ((net_exit - net_entry) / net_entry) * 100
            duration = len(df) - 1 - entry_idx
            trades.append({
                "Giriş Tarihi": entry_date,
                "Çıkış Tarihi": f"{index_arr[-1]} (açık)",
                "Giriş Fiyatı": round(entry_price, 2),
                "Çıkış Fiyatı": round(exit_price, 2),
                "Getiri (%)": round(pnl_pct, 2),
                "Süre (mum)": duration,
            })

        if trades:
            trades_df = pd.DataFrame(trades)

            # ---- Performans Metrikleri ----
            returns = trades_df["Getiri (%)"].values
            wins = returns[returns > 0]
            losses = returns[returns <= 0]

            total_trades = len(returns)
            win_count = len(wins)
            loss_count = len(losses)
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

            # Kümülatif getiri (bileşik)
            cumulative = 1.0
            for r in returns:
                cumulative *= (1 + r / 100)
            total_return = (cumulative - 1) * 100

            avg_win = float(wins.mean()) if len(wins) > 0 else 0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0
            best_trade = float(returns.max())
            worst_trade = float(returns.min())
            profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")

            # Equity Curve
            equity = [initial_capital]
            for r in returns:
                equity.append(equity[-1] * (1 + r / 100))

            peak = equity[0]
            drawdowns = []
            for e in equity:
                if e > peak:
                    peak = e
                dd = ((peak - e) / peak) * 100
                drawdowns.append(dd)
            max_drawdown = max(drawdowns)

            # Sharpe Ratio (basitleştirilmiş — trade bazlı)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns)) * np.sqrt(len(returns))
            else:
                sharpe = 0.0

            # Buy & Hold karşılaştırması
            first_close = float(close_arr[0])
            last_close_val = float(close_arr[-1])
            bh_return = ((last_close_val - first_close) / first_close) * 100

            # ---- Metrikler Gösterimi ----
            st.subheader("📈 Performans Özeti")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Toplam Getiri", f"%{total_return:.2f}")
            m2.metric("Buy & Hold", f"%{bh_return:.2f}")
            m3.metric("Win Rate", f"%{win_rate:.1f}")
            m4.metric("Max Drawdown", f"%{max_drawdown:.2f}")
            m5.metric("Sharpe Ratio", f"{sharpe:.2f}")

            m6, m7, m8, m9, m10 = st.columns(5)
            m6.metric("Toplam Trade", f"{total_trades}")
            m7.metric("Kazançlı", f"{win_count}")
            m8.metric("Zararlı", f"{loss_count}")
            m9.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞")
            m10.metric("Ort. Kazanç/Kayıp", f"{avg_win:.2f}% / {avg_loss:.2f}%")

            # ---- Strateji vs Buy & Hold karşılaştırması ----
            alpha = total_return - bh_return
            if alpha > 0:
                st.success(f"✅ Konsensüs stratejisi Buy & Hold'u **%{alpha:.2f}** geride bıraktı.")
            elif alpha < 0:
                st.error(f"❌ Konsensüs stratejisi Buy & Hold'un **%{abs(alpha):.2f}** gerisinde kaldı.")
            else:
                st.info("➡️ Konsensüs stratejisi Buy & Hold ile aynı performansı gösterdi.")

            # ---- Trade Tablosu ----
            st.subheader("📋 Trade Geçmişi")

            def trade_color(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "color: #00ff00"
                    elif val < 0:
                        return "color: #ff4b4b"
                return ""

            st.dataframe(
                trades_df.style.map(trade_color, subset=["Getiri (%)"]),
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------------
        # BİREYSEL ALGORİTMA PERFORMANS KARŞILAŞTIRMASI
        # (Konsensüs backtest'inden bağımsız, her zaman gösterilir)
        # -------------------------------------------------------
        st.write("---")
        st.header("🏆 Bireysel Algoritma Performansları")
        st.caption(
            "Her algoritmanın tek başına AL/SAT sinyallerini takip etmenin "
            "geçmiş performansı. Komisyon ve slippage dahildir."
        )

        algo_signal_map = {
            "SMA Crossover": "Sig_SMA",
            "RSI (14)": "Sig_RSI",
            "Bollinger Bands": "Sig_BB",
            "MACD": "Sig_MACD",
            "Mean Reversion": "Sig_Z",
            "OBV": "Sig_OBV",
            "ADX": "Sig_ADX",
            "Stoch RSI": "Sig_StochRSI",
            "Ichimoku": "Sig_Ichimoku",
        }
        if is_intraday:
            algo_signal_map["VWAP"] = "Sig_VWAP"

        def backtest_single_signal(signal_series):
            """Tek bir sinyal serisinin backtest'ini yapar."""
            sig_arr = signal_series.values
            s_trades = []
            s_in_pos = False
            s_entry = 0.0
            cost_pct = (commission_pct + slippage_pct) / 100

            for i in range(1, len(sig_arr)):
                if not s_in_pos and sig_arr[i] == 1 and sig_arr[i - 1] != 1:
                    s_entry = float(close_arr[i])
                    s_in_pos = True
                elif s_in_pos and sig_arr[i] == -1 and sig_arr[i - 1] != -1:
                    s_exit = float(close_arr[i])
                    net_e = s_entry * (1 + cost_pct)
                    net_x = s_exit * (1 - cost_pct)
                    pnl = ((net_x - net_e) / net_e) * 100
                    s_trades.append(pnl)
                    s_in_pos = False

            # Açık pozisyon → son fiyatla kapat
            if s_in_pos:
                s_exit = float(close_arr[-1])
                net_e = s_entry * (1 + cost_pct)
                net_x = s_exit * (1 - cost_pct)
                pnl = ((net_x - net_e) / net_e) * 100
                s_trades.append(pnl)

            if not s_trades:
                return {
                    "Trade": 0, "Getiri (%)": 0.0, "Win Rate (%)": 0.0,
                    "Ort. Kazanç (%)": 0.0, "Ort. Kayıp (%)": 0.0,
                    "Max DD (%)": 0.0, "Profit Factor": 0.0
                }

            returns_arr = np.array(s_trades)
            wins_arr = returns_arr[returns_arr > 0]
            losses_arr = returns_arr[returns_arr <= 0]

            cumul = 1.0
            peak_c = 1.0
            max_dd = 0.0
            for r in returns_arr:
                cumul *= (1 + r / 100)
                if cumul > peak_c:
                    peak_c = cumul
                dd = ((peak_c - cumul) / peak_c) * 100
                if dd > max_dd:
                    max_dd = dd

            total_ret = (cumul - 1) * 100
            wr = (len(wins_arr) / len(returns_arr)) * 100
            avg_w = float(wins_arr.mean()) if len(wins_arr) > 0 else 0.0
            avg_l = float(losses_arr.mean()) if len(losses_arr) > 0 else 0.0
            pf = abs(wins_arr.sum() / losses_arr.sum()) if len(losses_arr) > 0 and losses_arr.sum() != 0 else float("inf")

            return {
                "Trade": len(returns_arr),
                "Getiri (%)": round(total_ret, 2),
                "Win Rate (%)": round(wr, 1),
                "Ort. Kazanç (%)": round(avg_w, 2),
                "Ort. Kayıp (%)": round(avg_l, 2),
                "Max DD (%)": round(max_dd, 2),
                "Profit Factor": round(pf, 2) if pf != float("inf") else "∞"
            }

        algo_results = []
        for algo_name, sig_col in algo_signal_map.items():
            if sig_col in df.columns:
                result = backtest_single_signal(df[sig_col])
                result["Algoritma"] = algo_name
                algo_results.append(result)

        if algo_results:
            algo_df = pd.DataFrame(algo_results)
            # Algoritma sütununu başa al
            cols = ["Algoritma"] + [c for c in algo_df.columns if c != "Algoritma"]
            algo_df = algo_df[cols]

            # Trade'i olmayan algoritmaları filtrele
            active_algos = algo_df[algo_df["Trade"] > 0].copy()

            if not active_algos.empty:
                # En iyi algoritmayı bul (getiriye göre)
                # Profit Factor string olabilir (∞), sayısal karşılaştırma için getiri kullan
                best_idx = active_algos["Getiri (%)"].idxmax()
                best_algo = active_algos.loc[best_idx]

                st.success(
                    f"🥇 En iyi performans: **{best_algo['Algoritma']}** — "
                    f"Getiri: %{best_algo['Getiri (%)']}, "
                    f"Win Rate: %{best_algo['Win Rate (%)']}, "
                    f"{int(best_algo['Trade'])} trade"
                )

                # Tablo renklendirmesi
                def algo_return_color(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return "color: #00ff00"
                        elif val < 0:
                            return "color: #ff4b4b"
                    return ""

                st.dataframe(
                    algo_df.style.map(algo_return_color, subset=["Getiri (%)"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Hiçbir algoritma bu veri aralığında trade üretmedi.")
        else:
            st.info("Algoritma performansı hesaplanamadı.")

        if not trades:
            st.info(
                "Bu veri aralığında konsensüs eşiğini karşılayan trade sinyali bulunamadı. "
                "Eşiği düşürmeyi veya veri süresini artırmayı deneyin."
            )

        # -------------------------------------------------------
        # PARAMETRE OPTİMİZASYONU (Walk-Forward)
        # -------------------------------------------------------
        if run_optimization:
            st.write("---")
            st.header("🧬 Parametre Optimizasyonu")
            st.caption(
                "Walk-forward optimizasyon: Verinin ilk %70'inde en iyi parametre seti aranır, "
                "son %30'unda doğrulanır. Out-of-sample'da da iyi çalışıyorsa parametreler güvenilirdir."
            )

            opt_progress = st.progress(0, text="Optimizasyon başlatılıyor...")

            # Parametre aralıkları (daha dar tutuldu — hız için)
            param_grid = {
                "sma_s": [10, 15, 20, 25, 30],
                "sma_l": [50, 75, 100, 125, 150],
                "rsi_lower": [25, 30, 35],
                "rsi_upper": [65, 70, 75],
                "z_thresh": [1.5, 2.0, 2.5],
                "cons_thresh": [4, 5, 6, 7],
            }

            # Toplam kombinasyon sayısı
            from itertools import product as iter_product
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            all_combos = list(iter_product(*values))
            total_combos = len(all_combos)

            # Walk-forward split
            split_idx = int(len(df) * 0.7)
            df_in = df.iloc[:split_idx].copy()
            df_out = df.iloc[split_idx:].copy()
            close_in = close.iloc[:split_idx]
            close_out = close.iloc[split_idx:]

            def run_backtest_with_params(data_slice, close_slice, p_sma_s, p_sma_l,
                                          p_rsi_lower, p_rsi_upper, p_z_thresh, p_cons_thresh):
                """Verilen parametre setiyle konsensüs backtest çalıştırır."""
                d = data_slice.copy()
                c = close_slice.copy()
                h = d["High"].squeeze() if "High" in d.columns else c
                l = d["Low"].squeeze() if "Low" in d.columns else c
                v = d["Volume"].squeeze() if "Volume" in d.columns else pd.Series(0, index=c.index)

                # SMA
                sma_sh = c.rolling(window=p_sma_s, min_periods=p_sma_s).mean()
                sma_lo = c.rolling(window=p_sma_l, min_periods=p_sma_l).mean()
                sig_sma = np.where(sma_sh > sma_lo, 1, -1)
                sig_sma = np.where(sma_sh.isna() | sma_lo.isna(), 0, sig_sma)

                # RSI
                delta = c.diff()
                gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
                loss_s = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
                rs = gain / loss_s.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                sig_rsi = np.where(rsi < p_rsi_lower, 1, np.where(rsi > p_rsi_upper, -1, 0))

                # Bollinger
                mid = c.rolling(window=20).mean()
                std = c.rolling(window=20).std()
                up = mid + (std * 2)
                lo_bb = mid - (std * 2)
                sig_bb = np.where(c < lo_bb, 1, np.where(c > up, -1, 0))

                # MACD
                e12 = c.ewm(span=12, adjust=False).mean()
                e26 = c.ewm(span=26, adjust=False).mean()
                macd = e12 - e26
                macd_s = macd.ewm(span=9, adjust=False).mean()
                sig_macd = np.where(macd > macd_s, 1, -1)

                # Z-Score
                z_mean = c.rolling(30).mean()
                z_std = c.rolling(30).std().replace(0, np.nan)
                z_val = (c - z_mean) / z_std
                sig_z = np.where(z_val < -p_z_thresh, 1, np.where(z_val > p_z_thresh, -1, 0))

                # OBV
                obv_sign = np.sign(c.diff()).fillna(0)
                obv = (v * obv_sign).cumsum()
                obv_s = obv.rolling(window=10, min_periods=10).mean()
                obv_l = obv.rolling(window=30, min_periods=30).mean()
                sig_obv = np.where(obv_s > obv_l, 1, -1)
                sig_obv = np.where(obv_s.isna() | obv_l.isna(), 0, sig_obv)

                # ADX
                adx_v, pdi, mdi = calc_adx(h, l, c, period=14)
                sig_adx = np.where(adx_v > 25, np.where(pdi > mdi, 1, -1), 0)

                # Stoch RSI
                rsi_min = rsi.rolling(window=14, min_periods=14).min()
                rsi_max = rsi.rolling(window=14, min_periods=14).max()
                rsi_rng = (rsi_max - rsi_min).replace(0, np.nan)
                stoch_k = ((rsi - rsi_min) / rsi_rng) * 100
                sig_stoch = np.where(stoch_k < 20, 1, np.where(stoch_k > 80, -1, 0))

                # Ichimoku
                tenkan = (h.rolling(window=9).max() + l.rolling(window=9).min()) / 2
                kijun = (h.rolling(window=26).max() + l.rolling(window=26).min()) / 2
                senkou_a = ((tenkan + kijun) / 2).shift(26)
                senkou_b = ((h.rolling(window=52).max() + l.rolling(window=52).min()) / 2).shift(26)
                cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
                cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
                sig_ichi = np.where((tenkan > kijun) & (c > cloud_top), 1,
                                    np.where((tenkan < kijun) & (c < cloud_bot), -1, 0))

                # VWAP (intraday only)
                if is_intraday:
                    tp = (h + l + c) / 3
                    cv = v.cumsum()
                    vwap = (tp * v).cumsum() / cv.replace(0, np.nan)
                    sig_vwap = np.where(c > vwap, 1, -1)
                    sig_vwap = np.where(vwap.isna(), 0, sig_vwap)
                else:
                    sig_vwap = np.zeros(len(c))

                # Konsensüs
                all_sigs = np.column_stack([
                    sig_sma, sig_rsi, sig_bb, sig_macd, sig_z,
                    sig_obv, sig_adx, sig_vwap, sig_stoch, sig_ichi
                ])
                al_count = (all_sigs == 1).sum(axis=1)
                sat_count = (all_sigs == -1).sum(axis=1)
                consensus = np.zeros(len(c))
                consensus[al_count >= p_cons_thresh] = 1
                consensus[sat_count >= p_cons_thresh] = -1

                # Backtest
                cost = (commission_pct + slippage_pct) / 100
                c_arr = c.values
                bt_trades = []
                in_pos = False
                entry_p = 0.0

                for i in range(1, len(consensus)):
                    if not in_pos and consensus[i] == 1 and consensus[i - 1] != 1:
                        entry_p = float(c_arr[i])
                        in_pos = True
                    elif in_pos and consensus[i] == -1 and consensus[i - 1] != -1:
                        exit_p = float(c_arr[i])
                        net_e = entry_p * (1 + cost)
                        net_x = exit_p * (1 - cost)
                        bt_trades.append(((net_x - net_e) / net_e) * 100)
                        in_pos = False

                if in_pos:
                    exit_p = float(c_arr[-1])
                    net_e = entry_p * (1 + cost)
                    net_x = exit_p * (1 - cost)
                    bt_trades.append(((net_x - net_e) / net_e) * 100)

                if not bt_trades:
                    return 0.0, 0.0, 0, 0.0

                returns_arr = np.array(bt_trades)
                cumul = 1.0
                peak_c = 1.0
                max_dd = 0.0
                for r in returns_arr:
                    cumul *= (1 + r / 100)
                    if cumul > peak_c:
                        peak_c = cumul
                    dd = ((peak_c - cumul) / peak_c) * 100
                    if dd > max_dd:
                        max_dd = dd

                total_ret = (cumul - 1) * 100
                wr = (len(returns_arr[returns_arr > 0]) / len(returns_arr)) * 100
                sharpe_r = 0.0
                if len(returns_arr) > 1 and np.std(returns_arr) > 0:
                    sharpe_r = float(np.mean(returns_arr) / np.std(returns_arr)) * np.sqrt(len(returns_arr))

                return total_ret, sharpe_r, len(bt_trades), max_dd

            # Grid search çalıştır
            best_sharpe = -999
            best_params = {}
            best_in_result = {}
            results_list = []

            for idx, combo in enumerate(all_combos):
                p = dict(zip(keys, combo))

                # Minimum veri kontrolü
                if len(close_in) < max(p["sma_l"], 52) + 30:
                    continue

                ret, sharpe_r, n_trades, mdd = run_backtest_with_params(
                    df_in, close_in,
                    p["sma_s"], p["sma_l"], p["rsi_lower"], p["rsi_upper"],
                    p["z_thresh"], p["cons_thresh"]
                )

                results_list.append({**p, "Getiri": ret, "Sharpe": sharpe_r,
                                      "Trade": n_trades, "Max DD": mdd})

                if sharpe_r > best_sharpe and n_trades >= 2:
                    best_sharpe = sharpe_r
                    best_params = p.copy()
                    best_in_result = {"Getiri": ret, "Sharpe": sharpe_r,
                                      "Trade": n_trades, "Max DD": mdd}

                # Progress güncelle
                if idx % 10 == 0:
                    opt_progress.progress(
                        (idx + 1) / total_combos,
                        text=f"Test ediliyor: {idx + 1}/{total_combos}"
                    )

            opt_progress.progress(1.0, text="Optimizasyon tamamlandı!")

            if best_params:
                # Out-of-sample doğrulama
                out_ret, out_sharpe, out_trades, out_mdd = run_backtest_with_params(
                    df_out, close_out,
                    best_params["sma_s"], best_params["sma_l"],
                    best_params["rsi_lower"], best_params["rsi_upper"],
                    best_params["z_thresh"], best_params["cons_thresh"]
                )

                st.subheader("🏆 En İyi Parametre Seti")

                # Parametre tablosu
                param_display = {
                    "Hızlı SMA": best_params["sma_s"],
                    "Yavaş SMA": best_params["sma_l"],
                    "RSI Alt": best_params["rsi_lower"],
                    "RSI Üst": best_params["rsi_upper"],
                    "Z-Score Eşik": best_params["z_thresh"],
                    "Konsensüs Eşik": best_params["cons_thresh"],
                }
                p1, p2, p3 = st.columns(3)
                p1.metric("Hızlı SMA", best_params["sma_s"])
                p1.metric("RSI Alt", best_params["rsi_lower"])
                p2.metric("Yavaş SMA", best_params["sma_l"])
                p2.metric("RSI Üst", best_params["rsi_upper"])
                p3.metric("Z-Score Eşik", best_params["z_thresh"])
                p3.metric("Konsensüs Eşik", best_params["cons_thresh"])

                # In-sample vs Out-of-sample karşılaştırma
                st.subheader("📊 Walk-Forward Doğrulama")
                wf1, wf2 = st.columns(2)

                with wf1:
                    st.markdown("**In-Sample (%70)**")
                    st.metric("Getiri", f"%{best_in_result['Getiri']:.2f}")
                    st.metric("Sharpe Ratio", f"{best_in_result['Sharpe']:.2f}")
                    st.metric("Trade Sayısı", f"{best_in_result['Trade']}")
                    st.metric("Max Drawdown", f"%{best_in_result['Max DD']:.2f}")

                with wf2:
                    st.markdown("**Out-of-Sample (%30)**")
                    st.metric("Getiri", f"%{out_ret:.2f}")
                    st.metric("Sharpe Ratio", f"{out_sharpe:.2f}")
                    st.metric("Trade Sayısı", f"{out_trades}")
                    st.metric("Max Drawdown", f"%{out_mdd:.2f}")

                # Overfitting değerlendirmesi
                if out_ret > 0 and out_sharpe > 0:
                    st.success(
                        "✅ Out-of-sample pozitif getiri ve Sharpe — parametreler güvenilir görünüyor."
                    )
                elif out_ret > 0:
                    st.warning(
                        "⚠️ Out-of-sample pozitif getiri ama düşük Sharpe — dikkatli kullanın."
                    )
                else:
                    st.error(
                        "❌ Out-of-sample negatif getiri — overfitting riski yüksek. "
                        "Bu parametrelere güvenmeyin."
                    )

                # Top 10 kombinasyon tablosu
                if results_list:
                    st.subheader("📋 En İyi 10 Kombinasyon (In-Sample)")
                    res_opt_df = pd.DataFrame(results_list)
                    res_opt_df = res_opt_df[res_opt_df["Trade"] >= 2]
                    res_opt_df = res_opt_df.sort_values("Sharpe", ascending=False).head(10)
                    res_opt_df.columns = [
                        "SMA Kısa", "SMA Uzun", "RSI Alt", "RSI Üst",
                        "Z Eşik", "Kons. Eşik", "Getiri (%)", "Sharpe",
                        "Trade", "Max DD (%)"
                    ]
                    res_opt_df["Getiri (%)"] = res_opt_df["Getiri (%)"].round(2)
                    res_opt_df["Sharpe"] = res_opt_df["Sharpe"].round(2)
                    res_opt_df["Max DD (%)"] = res_opt_df["Max DD (%)"].round(2)
                    st.dataframe(res_opt_df, use_container_width=True, hide_index=True)

            else:
                st.warning(
                    "Hiçbir kombinasyon yeterli trade üretemedi. "
                    "Veri süresini artırmayı deneyin."
                )

    else:
        st.error(
            "Veri çekilemedi. Lütfen Ticker'ın doğruluğunu veya "
            "İnternet bağlantınızı kontrol edin."
        )
