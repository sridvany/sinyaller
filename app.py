import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ============================================================
# 1. SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(page_title="Algo-Trader Pro v3.2", layout="wide")

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
    st.subheader("Konsensüs Ayarları")
    consensus_threshold = st.slider(
        "Minimum AL/SAT çoğunluğu (8 algoritmadan):", 3, 7, 5
    )

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

        # -------------------------------------------------------
        # KONSENSÜS HESABI (her satır için)
        # -------------------------------------------------------
        signal_cols = ["Sig_SMA", "Sig_RSI", "Sig_BB", "Sig_MACD", "Sig_Z", "Sig_OBV", "Sig_ADX"]

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
        tab1, tab2, tab3, tab4 = st.tabs(["RSI", "MACD", "ADX", "OBV"])

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

    else:
        st.error(
            "Veri çekilemedi. Lütfen Ticker'ın doğruluğunu veya "
            "İnternet bağlantınızı kontrol edin."
        )
