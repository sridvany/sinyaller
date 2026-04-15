import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from itertools import product as iter_product

# ============================================================
# 1. SAYFA KONFİGÜRASYONU
# ============================================================
st.set_page_config(page_title="tahmin.ai", layout="wide")

auto_refresh_on = st.sidebar.toggle("🔄 Canlı Yenileme", value=True)
if auto_refresh_on:
    st_autorefresh(interval=55 * 1000, key="terminal_refresh")

st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; }
    div[data-testid="stCaption"] { margin-top: -0.5rem; margin-bottom: -0.5rem; }
    h1 { margin-bottom: 0 !important; padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 BORSA TERMİNALİ")
st.caption("YATIRIM TAVSİYESİ İÇERMEZ. ARAŞTIRMA İÇİNDİR.")

# ============================================================
# SESSION STATE VARSAYILANLARI
# ============================================================
_defaults = {
    "sma_short":     20,
    "sma_long":      200,
    "rsi_period":    14,
    "rsi_lower":     30,
    "rsi_upper":     70,
    "bb_period":     20,
    "bb_std":        2.0,
    "macd_fast":     12,
    "macd_slow":     26,
    "macd_signal":   9,
    "z_period":      30,
    "z_thresh":      2.0,
    "adx_period":    14,
    "adx_threshold": 25,
    "st_period":     10,
    "st_multiplier": 3.0,
    "lrc_period":    50,
    "lrc_std_mult":  2.0,
    "wt_n1":         10,
    "wt_n2":         21,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# 2. YAN PANEL
# ============================================================
with st.sidebar:
    st.header("⚙️ Veri Ayarları")
    ticker = st.text_input("Ticker Sembolü:", "gc=f")

    period = st.selectbox(
        "Toplam Veri Süresi (Period):",
        options=["1d", "5d", "1mo", "6mo", "1y", "2y", "5y", "max"],
        index=4,
    )

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
    chart_type = st.radio("📊 Grafik Tipi:", ["Mum", "Çizgi"], horizontal=True)

    st.write("---")
    st.subheader("Sabit Parametreler")
    ss = st.session_state
    sma_short        = st.slider("SMA Kısa Periyot:",        5,   50,  value=ss["sma_short"])
    sma_long         = st.slider("SMA Uzun Periyot:",        50,  300, value=ss["sma_long"])
    rsi_period       = st.slider("RSI Periyodu:",            7,   21,  value=ss["rsi_period"])
    rsi_lower        = st.slider("RSI Alt Eşik:",            20,  40,  value=ss["rsi_lower"])
    rsi_upper        = st.slider("RSI Üst Eşik:",            60,  80,  value=ss["rsi_upper"])
    rsi_ma_period    = st.slider("RSI MA Periyodu:",         5,   50,  14)
    bb_period        = st.slider("BB Periyodu:",             10,  50,  value=ss["bb_period"])
    bb_std           = st.slider("BB Standart Sapma:",       1.0, 3.0, value=ss["bb_std"],        step=0.5)
    macd_fast        = st.slider("MACD Hızlı EMA:",          5,   20,  value=ss["macd_fast"])
    macd_slow        = st.slider("MACD Yavaş EMA:",          15,  40,  value=ss["macd_slow"])
    macd_signal      = st.slider("MACD Sinyal:",             5,   15,  value=ss["macd_signal"])
    z_period         = st.slider("Z-Score Pencere:",         10,  60,  value=ss["z_period"])
    z_thresh         = st.slider("Z-Score Eşik:",            1.0, 3.0, value=ss["z_thresh"],      step=0.5)
    obv_short        = st.slider("OBV Kısa SMA:",            5,   20,  10)
    obv_long         = st.slider("OBV Uzun SMA:",            15,  50,  30)
    adx_period       = st.slider("ADX Periyodu:",            7,   30,  value=ss["adx_period"])
    adx_threshold    = st.slider("ADX Trend Eşiği:",        15,  35,  value=ss["adx_threshold"])
    atr_period       = st.slider("ATR Periyodu:",            7,   30,  14)
    stoch_rsi_period = st.slider("Stoch RSI Periyodu:",      7,   21,  14)
    stoch_d_period   = st.slider("Stoch RSI %D Smoothing:",  2,   5,   3)
    stoch_lower      = st.slider("Stoch RSI Alt Eşik:",      5,   30,  20)
    stoch_upper      = st.slider("Stoch RSI Üst Eşik:",      70,  95,  80)
    ichi_tenkan      = st.slider("Tenkan-sen:",              5,   20,  9)
    ichi_kijun       = st.slider("Kijun-sen:",               20,  40,  26)
    ichi_senkou_b    = st.slider("Senkou Span B:",           40,  65,  52)
    st_period        = st.slider("SuperTrend ATR Periyodu:", 5,   20,  value=ss["st_period"])
    st_multiplier    = st.slider("SuperTrend Çarpan:",       1.0, 5.0, value=ss["st_multiplier"], step=0.5)
    kama_period      = st.slider("KAMA Etkinlik Periyodu:",  5,   20,  10)
    kama_fast        = st.slider("KAMA Hızlı EMA:",          2,   5,   2)
    kama_slow        = st.slider("KAMA Yavaş EMA:",          20,  40,  30)
    lrc_period       = st.slider("LRC Periyodu:",            20,  100, value=ss["lrc_period"])
    lrc_std_mult     = st.slider("LRC Standart Sapma:",      1.0, 3.0, value=ss["lrc_std_mult"],  step=0.5)
    nw_bandwidth     = st.slider("NW Bant Genişliği (h):",   3,   20,  8)
    nw_window        = st.slider("NW Pencere (son N bar):",  50,  300, 100)
    vwap_band_pct    = st.slider("VWAP Nötr Bant (%):",     0.0, 1.0, 0.1, step=0.05)

    st.write("---")
    st.subheader("📐 Fibonacci Ayarları")
    fib_lookback = st.slider("Fibonacci Lookback (bar):", 20, 300, 100)

    st.write("---")
    st.subheader("〰️ WaveTrend Ayarları")
    wt_n1 = st.slider("WaveTrend Kanal (n1):",    5,  20,  value=ss["wt_n1"])
    wt_n2 = st.slider("WaveTrend Ortalama (n2):", 10, 40,  value=ss["wt_n2"])
    wt_ob = st.slider("WaveTrend Aşırı Alım:",    40, 80,  60)
    wt_os = st.slider("WaveTrend Aşırı Satım:",  -80, -20, -60)

    st.write("---")
    st.subheader("🔀 Divergence Ayarları")
    div_window = st.slider("Divergence Pivot Pencere:", 3, 10, 5)

    # ── Destek/Direnç ve Trend Çizgisi Ayarları ───────────────
    st.write("---")
    st.subheader("📊 Destek / Direnç Ayarları")
    swing_window  = st.slider("S/R Pivot Pencere:",    5,  20, 10,
        help="Tepe/dip tespiti için her yönde bakılacak bar sayısı")
    swing_touches = st.slider("Min. Dokunuş Sayısı:", 2,   5,  2,
        help="Bir seviyenin geçerli sayılması için minimum dokunuş")
    swing_tol     = st.slider("Tolerans (%):",        0.1, 1.0, 0.3, step=0.1,
        help="İki seviyenin aynı sayılması için maksimum fiyat farkı") / 100

    st.write("---")
    st.subheader("📐 Trend Çizgisi Ayarları")
    tl_pivot_window = st.slider("TL Pivot Pencere:",       5,  20,  10,
        help="Trend çizgisi pivot tespiti için pencere genişliği")
    tl_max_lines    = st.slider("Max Çizgi Sayısı:",       1,   5,   3,
        help="Her yönde (destek/direnç) gösterilecek maksimum çizgi")
    tl_tolerance    = st.slider("TL Tolerans (%):",        0.3, 2.0, 1.2, step=0.1,
        help="Pivotun çizgiye dokundu sayılması için fiyat toleransı") / 100
    tl_show_channel = st.checkbox("Kanalları Göster", value=True,
        help="Paralel destek+direnç kanallarını dolgulu göster")
    # ──────────────────────────────────────────────────────────

    st.write("---")
    st.subheader("📊 Backtest Ayarları")
    commission_pct = st.slider("Komisyon (% / işlem):", 0.0, 1.0, 0.1, step=0.01)
    slippage_pct   = st.slider("Slippage (% / işlem):", 0.0, 0.5, 0.05, step=0.01)

    st.write("---")
    st.subheader("🔁 Walk-Forward Optimizasyon")
    n_windows = st.slider("Pencere Sayısı:", 2, 8, 3,
        help="Veri kaç eşit parçaya bölünsün?")
    train_pct = st.slider("Eğitim Oranı (%):", 50, 85, 70, step=5,
        help="Her pencerenin %kaçı eğitim, kalanı test olsun?")
    st.caption(f"{n_windows} pencere · %{train_pct} eğitim / %{100-train_pct} test")

    st.write("---")
    run_opt = st.button("🚀 Algoritmaları Optimize Et", use_container_width=True, type="primary")
    st.info("İpucu: 1 dakikalık analizler için Periyot: 5d, Mum Aralığı: 1m seçiniz.")


# ============================================================
# 3. OPTİMİZASYON PARAMETRE GRİDLERİ
# ============================================================
PARAM_GRIDS = {
    "SMA Crossover":  {"sma_s":         [5, 10, 20, 30],
                       "sma_l":         [50, 100, 150, 200]},
    "RSI":            {"rsi_period":    [10, 14, 21],
                       "rsi_lower":     [25, 30, 35],
                       "rsi_upper":     [65, 70, 75]},
    "Bollinger Bands":{"bb_period":     [15, 20, 30],
                       "bb_std":        [1.5, 2.0, 2.5]},
    "MACD":           {"macd_fast":     [8, 12, 16],
                       "macd_slow":     [20, 26, 30],
                       "macd_signal":   [7, 9, 12]},
    "Mean Reversion": {"z_period":      [20, 30, 50],
                       "z_thresh":      [1.5, 2.0, 2.5]},
    "ADX":            {"adx_period":    [10, 14, 20],
                       "adx_threshold": [20, 25, 30]},
    "SuperTrend":     {"st_period":     [7, 10, 14],
                       "st_multiplier": [2.0, 2.5, 3.0, 3.5]},
    "LR Channel":     {"lrc_period":    [30, 50, 75],
                       "lrc_std_mult":  [1.5, 2.0, 2.5]},
    "WaveTrend":      {"wt_n1":         [8, 10, 14],
                       "wt_n2":         [15, 21, 28]},
}


# ============================================================
# 4. YARDIMCI FONKSİYONLAR
# ============================================================
def safe_scalar(value):
    if isinstance(value, (pd.Series, np.ndarray)):
        return float(value.iloc[0]) if len(value) > 0 else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        unique_tickers = df.columns.get_level_values(1).unique()
        if len(unique_tickers) <= 1:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    return df


def calc_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm   = pd.Series(plus_dm,  index=high.index, dtype=float)
    minus_dm  = pd.Series(minus_dm, index=high.index, dtype=float)
    alpha     = 1.0 / period
    atr_s     = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sp        = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    sm        = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di   = 100 * (sp / atr_s.replace(0, np.nan))
    minus_di  = 100 * (sm / atr_s.replace(0, np.nan))
    dx        = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx       = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def calc_kama(close, period=10, fast=2, slow=30):
    ca   = close.values.astype(float)
    kama = np.full(len(ca), np.nan)
    kama[period - 1] = ca[period - 1]
    fsc = 2.0 / (fast + 1)
    ssc = 2.0 / (slow + 1)
    for i in range(period, len(ca)):
        direction  = abs(ca[i] - ca[i - period])
        volatility = np.sum(np.abs(np.diff(ca[i - period:i + 1])))
        er  = 0.0 if volatility == 0 else direction / volatility
        sc  = (er * (fsc - ssc) + ssc) ** 2
        kama[i] = kama[i - 1] + sc * (ca[i] - kama[i - 1])
    return pd.Series(kama, index=close.index)


def calc_supertrend(high, low, close, period=10, multiplier=3.0):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    hl2 = (high + low) / 2
    ub  = (hl2 + multiplier * atr).values.astype(float)
    lb  = (hl2 - multiplier * atr).values.astype(float)
    ca  = close.values.astype(float)
    ubf = ub.copy()
    lbf = lb.copy()
    direction  = np.ones(len(ca), dtype=float)
    supertrend = np.full(len(ca), np.nan)
    for i in range(1, len(ca)):
        if np.isnan(ubf[i-1]) or np.isnan(lbf[i-1]):
            ubf[i] = ub[i]
            lbf[i] = lb[i]
        else:
            ubf[i] = ub[i] if (ub[i] < ubf[i-1] or ca[i-1] > ubf[i-1]) else ubf[i-1]
            lbf[i] = lb[i] if (lb[i] > lbf[i-1] or ca[i-1] < lbf[i-1]) else lbf[i-1]
        if   ca[i] > ubf[i-1]: direction[i] = 1
        elif ca[i] < lbf[i-1]: direction[i] = -1
        else:                   direction[i] = direction[i-1]
        supertrend[i] = lbf[i] if direction[i] == 1 else ubf[i]
    return (pd.Series(supertrend, index=close.index), pd.Series(direction, index=close.index),
            pd.Series(lbf, index=close.index),        pd.Series(ubf, index=close.index))


def calc_linear_regression_channel(close, period=50, std_mult=2.0):
    n = len(close)
    mid   = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        y = close.values[i - period + 1:i + 1].astype(float)
        x = np.arange(period)
        sl, ic = np.polyfit(x, y, 1)
        yp  = sl * x + ic
        std = np.std(y - yp)
        mid[i]   = yp[-1]
        upper[i] = yp[-1] + std_mult * std
        lower[i] = yp[-1] - std_mult * std
    return (pd.Series(mid, index=close.index), pd.Series(upper, index=close.index),
            pd.Series(lower, index=close.index))


def calc_nadaraya_watson(close, bandwidth=8, window=100):
    n     = len(close)
    nwl   = np.full(n, np.nan)
    start = max(0, n - window)
    y = close.values[start:].astype(float)
    m = len(y)
    for i in range(m):
        w = np.exp(-((i - np.arange(m)) ** 2) / (2 * bandwidth ** 2))
        nwl[start + i] = np.sum(w * y) / np.sum(w)
    nws = pd.Series(nwl, index=close.index)
    mae = np.nanmean(np.abs(close.values[start:] - nwl[start:]))
    return nws, nws + 2 * mae, nws - 2 * mae


def calc_vwap_daily(high, low, close, volume):
    tp = (high + low + close) / 3
    dk = pd.Series(close.index.date, index=close.index)
    return (tp * volume).groupby(dk).cumsum() / volume.groupby(dk).cumsum().replace(0, np.nan)


# ── YENİ: Swing Destek/Direnç ─────────────────────────────────────────────────
def find_swing_levels(high, low, close, window=10, min_touches=2, tolerance=0.003):
    """
    Swing High/Low bazlı otomatik destek/direnç tespiti.
    Yakın seviyeler birleştirilir, dokunuş sayısına göre güç atanır.
    """
    n      = len(close)
    levels = []

    for i in range(window, n - window):
        if high.iloc[i] == high.iloc[i - window: i + window + 1].max():
            levels.append(("R", float(high.iloc[i]), i))
        if low.iloc[i] == low.iloc[i - window: i + window + 1].min():
            levels.append(("S", float(low.iloc[i]), i))

    merged = []
    used   = set()
    for idx, (typ, price, bar) in enumerate(levels):
        if idx in used:
            continue
        touches     = [price]
        touch_bars  = [bar]
        for jdx, (typ2, price2, bar2) in enumerate(levels):
            if jdx != idx and jdx not in used:
                if abs(price2 - price) / price < tolerance:
                    touches.append(price2)
                    touch_bars.append(bar2)
                    used.add(jdx)
        used.add(idx)
        avg_price  = float(np.mean(touches))
        last_touch = max(touch_bars)
        merged.append({
            "type":       typ,
            "price":      avg_price,
            "touches":    len(touches),
            "last_touch": last_touch,
        })

    merged = [m for m in merged if m["touches"] >= min_touches]
    merged = sorted(merged, key=lambda x: -x["touches"])[:15]
    return merged
# ──────────────────────────────────────────────────────────────────────────────


# ── YENİ: Diyagonal Trend Çizgileri ───────────────────────────────────────────
def find_trendlines(high, low, close, pivot_window=10, max_lines=3, tolerance=0.012):
    """
    Gelişmiş otomatik trend çizgisi tespiti.
    - Swing high/low pivotları tespit edilir
    - Her ikili kombinasyon için çizgi skoru hesaplanır
       (dokunuş sayısı + yenilik + ihlal cezası)
    - Benzer eğimli çizgiler tekilleştirilir
    - Paralel destek+direnç çiftleri kanal olarak işaretlenir
    Döndürür: (lines, channels)
      lines   : list of dict  {type, x0,y0,x1,y1,slope,touches,last_touch}
      channels: list of dict  {support, resistance}
    """
    n     = len(close)
    dates = close.index

    # Pivot tespiti
    pivot_highs, pivot_lows = [], []
    for i in range(pivot_window, n - pivot_window):
        if high.iloc[i] == high.iloc[i - pivot_window: i + pivot_window + 1].max():
            pivot_highs.append((i, float(high.iloc[i])))
        if low.iloc[i] == low.iloc[i - pivot_window: i + pivot_window + 1].min():
            pivot_lows.append((i, float(low.iloc[i])))

    def _score_line(p1, p2, pivots, violation_series):
        x1, y1 = p1;  x2, y2 = p2
        if x2 == x1: return 0, []
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        touches   = []
        violations = 0
        for xi in range(min(x1, x2), n):
            y_line = slope * xi + intercept
            y_act  = float(violation_series.iloc[xi])
            rel    = (y_act - y_line) / (abs(y_line) + 1e-9)
            # Dokunuş: pivot bu çizgiye yeterince yakın mı?
            for (px, py) in pivots:
                if px == xi and abs(py - y_line) / (abs(y_line) + 1e-9) < tolerance:
                    touches.append((xi, py))
            # İhlal: fiyat destek/direnç çizgisini kırdı mı?
            if slope >= 0 and rel < -tolerance * 3:   violations += 1
            if slope < 0  and rel >  tolerance * 3:   violations += 1
        score = len(touches) - violations * 0.5
        return score, touches

    def _best_lines(pivots, violation_series, line_type):
        if len(pivots) < 2:
            return []
        candidates = []
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                p1, p2 = pivots[i], pivots[j]
                score, touches = _score_line(p1, p2, pivots, violation_series)
                if score < 1.5 or len(touches) < 2:
                    continue
                x1, y1 = p1;  x2, y2 = p2
                slope     = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                y_end     = slope * (n - 1) + intercept
                last_bar  = max(t[0] for t in touches)
                candidates.append({
                    "type":       line_type,
                    "x0":         x1,         "y0": y1,
                    "x1":         n - 1,      "y1": y_end,
                    "slope":      slope,
                    "intercept":  intercept,
                    "touches":    len(touches),
                    "last_touch": last_bar,
                    "score":      score,
                })
        # Sırala: skor desc, yenilik desc
        candidates.sort(key=lambda c: (-c["score"], -c["last_touch"]))
        # Benzer eğimli çizgileri tekilleştir
        unique = []
        for c in candidates:
            dup = any(
                abs(c["slope"] - u["slope"]) / (abs(u["slope"]) + 1e-9) < 0.08
                for u in unique
            )
            if not dup:
                unique.append(c)
            if len(unique) >= max_lines:
                break
        return unique

    support_lines    = _best_lines(pivot_lows,  low,  "support")
    resistance_lines = _best_lines(pivot_highs, high, "resistance")

    # Kanal tespiti: yaklaşık paralel destek + direnç çiftleri
    channels = []
    for sl in support_lines:
        for rl in resistance_lines:
            sdiff = abs(sl["slope"] - rl["slope"]) / (abs(sl["slope"]) + 1e-9)
            if sdiff < 0.12:
                channels.append({"support": sl, "resistance": rl})

    return support_lines + resistance_lines, channels, dates
# ──────────────────────────────────────────────────────────────────────────────


# ============================================================
# FİBONACCİ, WAVETREND, DIVERGENCE
# ============================================================
def calc_fibonacci(high, low, lookback=100):
    recent_high = float(high.rolling(lookback, min_periods=1).max().iloc[-1])
    recent_low  = float(low.rolling(lookback, min_periods=1).min().iloc[-1])
    diff = recent_high - recent_low
    if diff == 0:
        return {}, recent_high, recent_low
    levels = {
        "0.0%":   recent_low,
        "23.6%":  recent_low + 0.236 * diff,
        "38.2%":  recent_low + 0.382 * diff,
        "50.0%":  recent_low + 0.500 * diff,
        "61.8%":  recent_low + 0.618 * diff,
        "78.6%":  recent_low + 0.786 * diff,
        "100.0%": recent_high,
    }
    return levels, recent_high, recent_low


def calc_wavetrend(high, low, close, n1=10, n2=21):
    ap  = (high + low + close) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d   = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci  = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def detect_divergence(price, indicator, window=5):
    n      = len(price)
    result = np.zeros(n)
    pv     = price.values.astype(float)
    iv     = indicator.values.astype(float)
    for i in range(window * 2, n):
        seg_p = pv[max(0, i - window * 4):i + 1]
        seg_i = iv[max(0, i - window * 4):i + 1]
        m     = len(seg_p)
        lows_p = []; lows_i = []
        for j in range(window, m - window):
            if seg_p[j] == np.min(seg_p[j - window:j + window + 1]):
                lows_p.append(seg_p[j])
                lows_i.append(seg_i[j])
        if len(lows_p) >= 2:
            if lows_p[-1] < lows_p[-2] and lows_i[-1] > lows_i[-2]:
                result[i] = 1
        highs_p = []; highs_i = []
        for j in range(window, m - window):
            if seg_p[j] == np.max(seg_p[j - window:j + window + 1]):
                highs_p.append(seg_p[j])
                highs_i.append(seg_i[j])
        if len(highs_p) >= 2:
            if highs_p[-1] > highs_p[-2] and highs_i[-1] < highs_i[-2]:
                result[i] = -1
    return pd.Series(result, index=price.index)


# ============================================================
# 5. SİNYAL FONKSİYONLARI
# ============================================================
def sig_sma(close, atr_high, sma_s=20, sma_l=100):
    sh  = close.rolling(sma_s, min_periods=sma_s).mean()
    sl  = close.rolling(sma_l, min_periods=sma_l).mean()
    sig = np.where(sh > sl, 1, -1)
    sig = np.where(sh.isna() | sl.isna(), 0, sig)
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), sh, sl


def sig_rsi_fn(close, rsi_period, rsi_lower=30, rsi_upper=70):
    d   = close.diff()
    g   = d.where(d > 0, 0.0).rolling(rsi_period).mean()
    l   = (-d.where(d < 0, 0.0)).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + g / l.replace(0, np.nan)))
    sig = np.where(rsi < rsi_lower, 1, np.where(rsi > rsi_upper, -1, 0))
    return pd.Series(sig, index=close.index), rsi


def sig_bb(close, bb_period, bb_std_val=2.0):
    mid = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    up  = mid + bb_std_val * std
    lo  = mid - bb_std_val * std
    sig = np.where(close < lo, 1, np.where(close > up, -1, 0))
    return pd.Series(sig, index=close.index), mid, up, lo


def sig_macd(close, atr_high, macd_fast=12, macd_slow=26, macd_sig_p=9):
    ef   = close.ewm(span=macd_fast, adjust=False).mean()
    es   = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ef - es
    ms   = macd.ewm(span=macd_sig_p, adjust=False).mean()
    sig  = np.where(macd > ms, 1, -1)
    sig  = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), macd, ms


def sig_z(close, z_period, z_thresh=2.0):
    zm  = close.rolling(z_period).mean()
    zs  = close.rolling(z_period).std().replace(0, np.nan)
    z   = (close - zm) / zs
    sig = np.where(z < -z_thresh, 1, np.where(z > z_thresh, -1, 0))
    return pd.Series(sig, index=close.index), z


def sig_obv(close, volume, obv_short, obv_long):
    obv = (volume * np.sign(close.diff()).fillna(0)).cumsum()
    s   = obv.rolling(obv_short, min_periods=obv_short).mean()
    l   = obv.rolling(obv_long,  min_periods=obv_long).mean()
    sig = np.where(s > l, 1, -1)
    sig = np.where(s.isna() | l.isna(), 0, sig)
    return pd.Series(sig, index=close.index), obv, s, l


def sig_adx_fn(high, low, close, atr_high, adx_period, adx_threshold=25):
    adx_v, pdi, mdi = calc_adx(high, low, close, period=adx_period)
    sig = np.where(adx_v > adx_threshold, np.where(pdi > mdi, 1, -1), 0)
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), adx_v, pdi, mdi


def sig_stochrsi(close, rsi_series, srsi_period, sd_period, sl, su):
    rmin = rsi_series.rolling(srsi_period, min_periods=srsi_period).min()
    rmax = rsi_series.rolling(srsi_period, min_periods=srsi_period).max()
    k    = ((rsi_series - rmin) / (rmax - rmin).replace(0, np.nan) * 100).fillna(50).clip(0, 100)
    d    = k.rolling(sd_period).mean()
    sig  = np.where(k < sl, 1, np.where(k > su, -1, 0))
    return pd.Series(sig, index=close.index), k, d


def sig_ichimoku(high, low, close, atr_high, it, ik, isb):
    tenkan   = (high.rolling(it).max()  + low.rolling(it).min())  / 2
    kijun    = (high.rolling(ik).max()  + low.rolling(ik).min())  / 2
    senkou_a = ((tenkan + kijun) / 2).shift(ik)
    senkou_b = ((high.rolling(isb).max() + low.rolling(isb).min()) / 2).shift(ik)
    ct = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cb = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    sig = np.where((tenkan > kijun) & (close > ct), 1,
                   np.where((tenkan < kijun) & (close < cb), -1, 0))
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), tenkan, kijun, senkou_a, senkou_b


def sig_kama_fn(close, atr_high, kp, kf, ks):
    kama = calc_kama(close, period=kp, fast=kf, slow=ks)
    sig  = np.where(close > kama, 1, np.where(close < kama, -1, 0))
    sig  = np.where(kama.isna(), 0, sig)
    sig  = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), kama


def sig_supertrend_fn(high, low, close, atr_high, stp, stm):
    st, std, lb, ub = calc_supertrend(high, low, close, period=stp, multiplier=stm)
    sig = std.values.copy()
    sig = np.where(st.isna(), 0, sig)
    sig = np.where(atr_high | (sig == 0), sig, 0)
    return pd.Series(sig, index=close.index), st, std, lb, ub


def sig_lrc(close, lrc_period, lrc_std_mult=2.0):
    mid, up, lo = calc_linear_regression_channel(close, period=lrc_period, std_mult=lrc_std_mult)
    sig = np.where(close < lo, 1, np.where(close > up, -1, 0))
    sig = np.where(mid.isna(), 0, sig)
    return pd.Series(sig, index=close.index), mid, up, lo


def sig_vwap_fn(high, low, close, volume, vwap_band_pct):
    vwap = calc_vwap_daily(high, low, close, volume)
    band = vwap * (vwap_band_pct / 100)
    sig  = np.where(close > vwap + band, 1, np.where(close < vwap - band, -1, 0))
    sig  = np.where(vwap.isna(), 0, sig)
    return pd.Series(sig, index=close.index), vwap


def sig_wavetrend_fn(high, low, close, n1=10, n2=21, ob=60, os_=-60):
    wt1, wt2   = calc_wavetrend(high, low, close, n1=n1, n2=n2)
    cross_up   = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    cross_down = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    sig = np.where(cross_up & (wt1 < os_), 1,
                   np.where(cross_down & (wt1 > ob), -1, 0))
    return pd.Series(sig, index=close.index), wt1, wt2


# ============================================================
# 6. BACKTEST YARDIMCISI
# ============================================================
def run_backtest(signal_series, close_arr, cost_pct):
    sig    = signal_series.values if hasattr(signal_series, "values") else signal_series
    trades = []
    in_pos = False
    entry_p = 0.0
    for i in range(1, len(sig)):
        if not in_pos and sig[i] == 1 and sig[i-1] != 1:
            entry_p = float(close_arr[i])
            in_pos  = True
        elif in_pos and sig[i] == -1 and sig[i-1] != -1:
            ep = float(close_arr[i])
            trades.append(((ep * (1 - cost_pct) - entry_p * (1 + cost_pct)) / (entry_p * (1 + cost_pct))) * 100)
            in_pos = False
    if in_pos:
        ep = float(close_arr[-1])
        trades.append(((ep * (1 - cost_pct) - entry_p * (1 + cost_pct)) / (entry_p * (1 + cost_pct))) * 100)
    if not trades:
        return {"total_ret": 0.0, "sharpe": 0.0, "n": 0,
                "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "max_dd": 0.0, "pf": 0.0}
    r      = np.array(trades)
    cumul  = 1.0
    peak   = 1.0
    max_dd = 0.0
    for rv in r:
        cumul *= (1 + rv / 100)
        if cumul > peak: peak = cumul
        dd = ((peak - cumul) / peak) * 100
        if dd > max_dd: max_dd = dd
    wins      = r[r > 0]
    losses    = r[r <= 0]
    total_ret = (cumul - 1) * 100
    wr        = len(wins) / len(r) * 100
    sharpe    = float(np.mean(r) / np.std(r)) * np.sqrt(len(r)) if len(r) > 1 and np.std(r) > 0 else 0.0
    pf        = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    return {"total_ret": round(total_ret, 4), "sharpe": round(sharpe, 4), "n": len(r),
            "win_rate": round(wr, 2),
            "avg_win":  round(float(wins.mean())   if len(wins)   > 0 else 0.0, 4),
            "avg_loss": round(float(losses.mean())  if len(losses) > 0 else 0.0, 4),
            "max_dd":   round(max_dd, 4),
            "pf":       round(pf, 4) if pf != float("inf") else float("inf")}


def _score(stats, metric):
    if metric == "Sharpe":
        return stats["sharpe"]
    elif metric == "Getiri":
        return stats["total_ret"]
    else:
        dd = stats["max_dd"]
        return stats["total_ret"] / dd if dd > 0 else stats["total_ret"]


def optimize_algo(param_grid, signal_fn, close_arr, cost_pct,
                  n_windows=4, train_pct=70, metric="Sharpe", min_trades=3):
    keys    = list(param_grid.keys())
    combos  = list(iter_product(*param_grid.values()))
    n       = len(close_arr)
    default = {k: v[0] for k, v in param_grid.items()}

    win_size = n // n_windows
    if win_size < 30:
        n_windows = 1
        win_size  = n

    windows = []
    for w in range(n_windows):
        s     = w * win_size
        e     = s + win_size if w < n_windows - 1 else n
        split = s + int((e - s) * train_pct / 100)
        if split - s < 10 or e - split < 5:
            continue
        windows.append((s, split, e))

    if not windows:
        return default, None

    combo_scores = {combo: [] for combo in combos}

    for (ts, te, es) in windows:
        test_arr = close_arr[te:es]
        for combo in combos:
            p        = dict(zip(keys, combo))
            sig_full = signal_fn(p)
            if sig_full is None:
                continue
            sig_vals   = sig_full.values if hasattr(sig_full, "values") else sig_full
            test_sig   = sig_vals[te:es]
            test_stats = run_backtest(test_sig, test_arr, cost_pct)
            if test_stats["n"] < min_trades:
                continue
            combo_scores[combo].append(_score(test_stats, metric))

    best_combo = None
    best_avg   = -np.inf
    for combo, scores in combo_scores.items():
        if not scores:
            continue
        avg = float(np.mean(scores))
        if avg > best_avg:
            best_avg   = avg
            best_combo = combo

    if best_combo is None:
        return default, None

    best_p   = dict(zip(keys, best_combo))
    sig_full = signal_fn(best_p)
    if sig_full is None:
        return best_p, None
    best_s = run_backtest(sig_full, close_arr, cost_pct)
    best_s["wf_avg_score"] = round(best_avg, 4)
    best_s["wf_windows"]   = len(windows)
    return best_p, best_s


# ============================================================
# 7. VERİ ÇEKME
# ============================================================
@st.cache_data(ttl=55)
def fetch_live_data(symbol, p, i):
    try:
        data = yf.download(symbol, period=p, interval=i, progress=False)
        return pd.DataFrame() if data is None or data.empty else data
    except Exception as e:
        st.error(f"Veri çekme hatası: {e}")
        return pd.DataFrame()


PLOTLY_CONFIG = dict(scrollZoom=True, displayModeBar=True,
    modeBarButtonsToAdd=["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
    modeBarButtonsToRemove=["lasso2d", "select2d"])


def sub_layout(height=250):
    return dict(template="plotly_dark", height=height, margin=dict(t=30, b=30), dragmode="pan")


# ============================================================
# 8. ANA MANTIK
# ============================================================
if ticker:
    df = fetch_live_data(ticker, period, interval)

    if not df.empty:
        df = flatten_columns(df)
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in df.columns]
        if missing:
            st.error(f"Eksik sütunlar: {missing}.")
            st.stop()

        close     = df["Close"].squeeze()
        high      = df["High"].squeeze()
        low       = df["Low"].squeeze()
        volume    = df["Volume"].squeeze()
        close_arr = close.values
        n_bars    = len(close)

        indicator_min_reqs = {
            "SMA Crossover":    sma_long,
            "Bollinger Bands":  bb_period,
            "RSI":              rsi_period * 2,
            "MACD":             macd_slow + macd_signal,
            "Mean Reversion":   z_period,
            "OBV":              obv_long,
            "ADX":              adx_period * 3,
            "Stoch RSI":        rsi_period + stoch_rsi_period,
            "Ichimoku":         ichi_senkou_b + ichi_kijun,
            "KAMA":             kama_period + kama_slow,
            "SuperTrend":       st_period * 2,
            "LR Channel":       lrc_period,
            "WaveTrend":        wt_n1 + wt_n2,
            "Walk-Forward Opt": 150,
        }

        affected = [
            f"{name} (min {req} mum)"
            for name, req in indicator_min_reqs.items()
            if n_bars < req
        ]

        min_req = max(150, adx_period * 3, ichi_senkou_b)
        if n_bars < min_req:
            if affected:
                st.warning(
                    f"⚠️ Yeterli veri yok: **{n_bars} mum** mevcut, en az **{min_req}** gerekli.\n\n"
                    f"**Etkilenen indikatörler:** {', '.join(affected)}"
                )
            else:
                st.warning(f"Yeterli veri yok: {n_bars} mum, en az {min_req} gerekli.")

        cost_pct    = (commission_pct + slippage_pct) / 100
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"]

        # ATR
        tr1        = high - low
        tr2        = (high - close.shift(1)).abs()
        tr3        = (low  - close.shift(1)).abs()
        tr         = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()
        atr_ma     = atr_series.rolling(atr_period, min_periods=atr_period).mean()
        atr_high   = (atr_series > atr_ma).values

        # ── YENİ: 200 EMA ─────────────────────────────────────────
        df["EMA200"] = close.ewm(span=200, adjust=False).mean()
        # ──────────────────────────────────────────────────────────

        # ── Swing Destek/Direnç (yatay) ───────────────────────────
        swing_levels = find_swing_levels(
            high, low, close,
            window=swing_window,
            min_touches=swing_touches,
            tolerance=swing_tol,
        )

        # ── Diyagonal Trend Çizgileri ──────────────────────────────
        trendlines, tl_channels, tl_dates = find_trendlines(
            high, low, close,
            pivot_window=tl_pivot_window,
            max_lines=tl_max_lines,
            tolerance=tl_tolerance,
        )
        # ──────────────────────────────────────────────────────────

        # ============================================================
        # OPTİMİZASYON
        # ============================================================
        OPT_KEY = f"opt_{ticker}_{period}_{interval}_{n_windows}_{train_pct}"

        if run_opt or OPT_KEY not in st.session_state:
            opt_params = {}
            opt_stats  = {}
            prog       = st.progress(0, text="Optimizasyon başlatılıyor…")
            algo_list  = list(PARAM_GRIDS.keys())

            for idx, algo_name in enumerate(algo_list):
                prog.progress(idx / len(algo_list), text=f"Optimize ediliyor: {algo_name}")
                grid = PARAM_GRIDS[algo_name]

                if algo_name == "SMA Crossover":
                    def make_fn():
                        def fn(p):
                            if p["sma_s"] >= p["sma_l"]: return None
                            s, _, _ = sig_sma(close, atr_high, p["sma_s"], p["sma_l"]); return s
                        return fn
                elif algo_name == "RSI":
                    def make_fn():
                        def fn(p):
                            if p["rsi_lower"] >= p["rsi_upper"]: return None
                            s, _ = sig_rsi_fn(close, p["rsi_period"], p["rsi_lower"], p["rsi_upper"]); return s
                        return fn
                elif algo_name == "Bollinger Bands":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_bb(close, p["bb_period"], p["bb_std"]); return s
                        return fn
                elif algo_name == "MACD":
                    def make_fn():
                        def fn(p):
                            if p["macd_fast"] >= p["macd_slow"]: return None
                            s, _, _ = sig_macd(close, atr_high, p["macd_fast"], p["macd_slow"], p["macd_signal"]); return s
                        return fn
                elif algo_name == "Mean Reversion":
                    def make_fn():
                        def fn(p):
                            s, _ = sig_z(close, p["z_period"], p["z_thresh"]); return s
                        return fn
                elif algo_name == "ADX":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_adx_fn(high, low, close, atr_high, p["adx_period"], p["adx_threshold"]); return s
                        return fn
                elif algo_name == "SuperTrend":
                    def make_fn():
                        def fn(p):
                            s, _, _, _, _ = sig_supertrend_fn(high, low, close, atr_high, p["st_period"], p["st_multiplier"]); return s
                        return fn
                elif algo_name == "LR Channel":
                    def make_fn():
                        def fn(p):
                            s, _, _, _ = sig_lrc(close, p["lrc_period"], p["lrc_std_mult"]); return s
                        return fn
                elif algo_name == "WaveTrend":
                    def make_fn():
                        def fn(p):
                            s, _, _ = sig_wavetrend_fn(high, low, close, p["wt_n1"], p["wt_n2"], wt_ob, wt_os); return s
                        return fn

                best_p, best_s = optimize_algo(
                    grid, make_fn(), close_arr, cost_pct,
                    n_windows=n_windows, train_pct=train_pct,
                    metric="Sharpe", min_trades=1)
                opt_params[algo_name] = best_p
                opt_stats[algo_name]  = best_s if best_s else {"total_ret": 0.0, "sharpe": 0.0, "n": 0, "win_rate": 0.0}

            prog.progress(1.0, text="✅ Optimizasyon tamamlandı!")
            st.session_state[OPT_KEY] = {"params": opt_params, "stats": opt_stats}

            p = opt_params
            st.session_state["sma_short"]     = int(p["SMA Crossover"]["sma_s"])
            st.session_state["sma_long"]      = int(p["SMA Crossover"]["sma_l"])
            st.session_state["rsi_period"]    = int(p["RSI"]["rsi_period"])
            st.session_state["rsi_lower"]     = int(p["RSI"]["rsi_lower"])
            st.session_state["rsi_upper"]     = int(p["RSI"]["rsi_upper"])
            st.session_state["bb_period"]     = int(p["Bollinger Bands"]["bb_period"])
            st.session_state["bb_std"]        = float(p["Bollinger Bands"]["bb_std"])
            st.session_state["macd_fast"]     = int(p["MACD"]["macd_fast"])
            st.session_state["macd_slow"]     = int(p["MACD"]["macd_slow"])
            st.session_state["macd_signal"]   = int(p["MACD"]["macd_signal"])
            st.session_state["z_period"]      = int(p["Mean Reversion"]["z_period"])
            st.session_state["z_thresh"]      = float(p["Mean Reversion"]["z_thresh"])
            st.session_state["adx_period"]    = int(p["ADX"]["adx_period"])
            st.session_state["adx_threshold"] = int(p["ADX"]["adx_threshold"])
            st.session_state["st_period"]     = int(p["SuperTrend"]["st_period"])
            st.session_state["st_multiplier"] = float(p["SuperTrend"]["st_multiplier"])
            st.session_state["lrc_period"]    = int(p["LR Channel"]["lrc_period"])
            st.session_state["lrc_std_mult"]  = float(p["LR Channel"]["lrc_std_mult"])
            st.session_state["wt_n1"]         = int(p["WaveTrend"]["wt_n1"])
            st.session_state["wt_n2"]         = int(p["WaveTrend"]["wt_n2"])
            st.rerun()

        else:
            opt_params = st.session_state[OPT_KEY]["params"]
            opt_stats  = st.session_state[OPT_KEY]["stats"]

        p_sma  = {"sma_s": sma_short,   "sma_l": sma_long}
        p_rsi  = {"rsi_period": rsi_period, "rsi_lower": rsi_lower, "rsi_upper": rsi_upper}
        p_bb   = {"bb_period": bb_period,   "bb_std": bb_std}
        p_macd = {"macd_fast": macd_fast,   "macd_slow": macd_slow, "macd_signal": macd_signal}
        p_z    = {"z_period": z_period,     "z_thresh": z_thresh}
        p_adx  = {"adx_period": adx_period, "adx_threshold": adx_threshold}
        p_st   = {"st_period": st_period,   "st_multiplier": st_multiplier}
        p_lrc  = {"lrc_period": lrc_period, "lrc_std_mult": lrc_std_mult}
        p_wt   = {"wt_n1": wt_n1,           "wt_n2": wt_n2}

        df["Sig_SMA"], df["SMA_SHORT"], df["SMA_LONG"] = sig_sma(
            close, atr_high, p_sma["sma_s"], p_sma["sma_l"])

        df["Sig_RSI"], df["RSI"] = sig_rsi_fn(
            close, p_rsi["rsi_period"], p_rsi["rsi_lower"], p_rsi["rsi_upper"])
        df["RSI_MA"] = df["RSI"].rolling(rsi_ma_period).mean()

        df["Sig_BB"], df["Mid"], df["Up"], df["Low_BB"] = sig_bb(
            close, p_bb["bb_period"], p_bb["bb_std"])

        df["Sig_MACD"], df["MACD"], df["MACD_S"] = sig_macd(
            close, atr_high, p_macd["macd_fast"], p_macd["macd_slow"], p_macd["macd_signal"])

        df["Sig_Z"], df["Z"] = sig_z(close, p_z["z_period"], p_z["z_thresh"])

        df["Sig_OBV"], df["OBV"], obv_sma_short, obv_sma_long = sig_obv(
            close, volume, obv_short, obv_long)

        df["Sig_ADX"], df["ADX"], df["PLUS_DI"], df["MINUS_DI"] = sig_adx_fn(
            high, low, close, atr_high, p_adx["adx_period"], p_adx["adx_threshold"])

        df["Sig_StochRSI"], df["StochRSI_K"], df["StochRSI_D"] = sig_stochrsi(
            close, df["RSI"], stoch_rsi_period, stoch_d_period, stoch_lower, stoch_upper)

        df["Sig_Ichimoku"], df["Tenkan"], df["Kijun"], df["Senkou_A"], df["Senkou_B"] = sig_ichimoku(
            high, low, close, atr_high, ichi_tenkan, ichi_kijun, ichi_senkou_b)

        df["Sig_KAMA"], df["KAMA"] = sig_kama_fn(
            close, atr_high, kama_period, kama_fast, kama_slow)

        df["Sig_SuperTrend"], df["SuperTrend"], df["ST_Direction"], df["ST_Lower"], df["ST_Upper"] = sig_supertrend_fn(
            high, low, close, atr_high, p_st["st_period"], p_st["st_multiplier"])

        df["Sig_LRC"], df["LRC_Mid"], df["LRC_Upper"], df["LRC_Lower"] = sig_lrc(
            close, p_lrc["lrc_period"], p_lrc["lrc_std_mult"])

        df["NW_Line"], df["NW_Upper"], df["NW_Lower"] = calc_nadaraya_watson(
            close, bandwidth=nw_bandwidth, window=nw_window)

        df["ATR"]      = atr_series
        df["ATR_High"] = atr_high

        if is_intraday:
            df["Sig_VWAP"], df["VWAP"] = sig_vwap_fn(high, low, close, volume, vwap_band_pct)
        else:
            df["Sig_VWAP"] = 0
            df["VWAP"]     = np.nan

        df["Sig_WaveTrend"], df["WT1"], df["WT2"] = sig_wavetrend_fn(
            high, low, close, p_wt["wt_n1"], p_wt["wt_n2"], wt_ob, wt_os)

        fib_levels, fib_high, fib_low = calc_fibonacci(high, low, lookback=fib_lookback)

        df["Div_RSI"]  = detect_divergence(close, df["RSI"],  window=div_window)
        df["Div_MACD"] = detect_divergence(close, df["MACD"], window=div_window)

        # ============================================================
        # ANA GRAFİK + VRP
        # ============================================================
        from plotly.subplots import make_subplots

        bull_st = df["ST_Direction"] == 1
        bear_st = df["ST_Direction"] == -1

        st_dir_shifted = df["ST_Direction"].shift(1).fillna(0)
        st_buy_signal  = (df["ST_Direction"] == 1)  & (st_dir_shifted != 1)
        st_sell_signal = (df["ST_Direction"] == -1) & (st_dir_shifted != -1)

        lp = float(close.iloc[-1])
        pp = float(close.iloc[-2]) if len(close) > 1 else lp

        vrp_bins     = 40
        price_min    = float(low.min())
        price_max    = float(high.max())
        bin_edges    = np.linspace(price_min, price_max, vrp_bins + 1)
        bin_centers  = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_at_price = np.zeros(vrp_bins)
        for i in range(len(df)):
            lo_i  = float(low.iloc[i])
            hi_i  = float(high.iloc[i])
            vol_i = float(volume.iloc[i])
            if hi_i == lo_i:
                idx = np.clip(np.searchsorted(bin_edges, lo_i, side="right") - 1, 0, vrp_bins - 1)
                vol_at_price[idx] += vol_i
            else:
                for b in range(vrp_bins):
                    overlap = min(hi_i, bin_edges[b+1]) - max(lo_i, bin_edges[b])
                    if overlap > 0:
                        vol_at_price[b] += vol_i * overlap / (hi_i - lo_i)

        poc_idx   = int(np.argmax(vol_at_price))
        poc_price = bin_centers[poc_idx]
        max_vol   = vol_at_price.max()
        bar_colors = [
            "rgba(255,165,0,1.0)" if b == poc_idx
            else f"rgba(100,{int(80 + 175*(v/max_vol)) if max_vol > 0 else 200},255,0.85)"
            for b, v in enumerate(vol_at_price)
        ]

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.75, 0.25],
            shared_yaxes=True,
            horizontal_spacing=0.0,
        )

        if chart_type == "Mum":
            # ── Sinyal bazlı mum renklendirme ─────────────────────
            cyan_raw   = (df["ST_Direction"] == 1) & (df["Sig_OBV"] == 1) & (df["RSI"] < 70)
            cyan_mask  = cyan_raw & ~cyan_raw.shift(1).fillna(False)
            yellow_mask = (~cyan_mask) & (df["ADX"] < adx_threshold) & (df["RSI"] >= 45) & (df["RSI"] <= 55)
            red_mask   = (~cyan_mask) & (~yellow_mask) & (df["Close"] < df["Open"]) & (df["MACD"] < df["MACD_S"])
            green_mask = ~cyan_mask & ~yellow_mask & ~red_mask

            _color_defs = [
                ("Cyan AL",  cyan_mask,   "#00ffff"),
                ("Yeşil",    green_mask,  "#00cc66"),
                ("Sarı",     yellow_mask, "#ffcc00"),
                ("Ayı",      red_mask,    "#ff4444"),
            ]
            for _lbl, _mask, _color in _color_defs:
                _rising  = _mask & (df["Close"] >= df["Open"])
                _falling = _mask & (df["Close"] <  df["Open"])
                for _m, _fill, _trace_lbl in [
                    (_rising,  _color,   _lbl + " ↑"),
                    (_falling, "#111111", _lbl + " ↓"),
                ]:
                    if _m.any():
                        fig.add_trace(go.Candlestick(
                            x=df.index[_m],
                            open=df["Open"][_m], high=df["High"][_m],
                            low=df["Low"][_m],   close=df["Close"][_m],
                            name=_trace_lbl,
                            increasing_fillcolor=_fill, increasing_line_color=_color,
                            decreasing_fillcolor=_fill, decreasing_line_color=_color,
                            showlegend=False,
                        ), row=1, col=1)

            # ── Divergence marker katmanı (ana grafik) ────────────
            bull_div = (df["Div_RSI"] == 1) | (df["Div_MACD"] == 1)
            bear_div = (df["Div_RSI"] == -1) | (df["Div_MACD"] == -1)
            if bull_div.any():
                fig.add_trace(go.Scatter(
                    x=df.index[bull_div], y=df["Low"][bull_div] * 0.998,
                    mode="markers", name="Bullish Div 🔺",
                    marker=dict(symbol="triangle-up", color="lime", size=10),
                ), row=1, col=1)
            if bear_div.any():
                fig.add_trace(go.Scatter(
                    x=df.index[bear_div], y=df["High"][bear_div] * 1.002,
                    mode="markers", name="Bearish Div 🔻",
                    marker=dict(symbol="triangle-down", color="red", size=10),
                ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat",
                line=dict(color="orange", width=1.5)), row=1, col=1)

        # ── Mum renk legend girişleri (dummy scatter) ─────────────
        if chart_type == "Mum":
            for _leg_name, _leg_color in [
                ("🔴 Ayı",         "#ff4444"),
                ("🟡 Kararsız",    "#ffcc00"),
                ("🟢 Boğa",        "#00cc66"),
                ("🔵 Güçlü Boğa",  "#00ffff"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=_leg_name,
                    marker=dict(symbol="square", size=16, color=_leg_color),
                    showlegend=True,
                ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_SHORT"],
            name=f"SMA {p_sma['sma_s']}", visible="legendonly",
            line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_LONG"],
            name=f"SMA {p_sma['sma_l']}", visible="legendonly",
            line=dict(color="cyan")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["KAMA"],
            name="KAMA", line=dict(color="violet", width=1.5)), row=1, col=1)

        # ── YENİ: 200 EMA trace ───────────────────────────────────
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"],
            name="EMA 200",
            line=dict(color="yellow", width=2, dash="dot"),
            visible="legendonly",
        ), row=1, col=1)
        # ──────────────────────────────────────────────────────────

        fig.add_trace(go.Scatter(
            x=df.index[bull_st], y=df["SuperTrend"][bull_st],
            name="SuperTrend (Boğa çizgi)", mode="lines",
            line=dict(color="rgba(0,255,100,0.5)", width=1.5),
            visible="legendonly", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index[bear_st], y=df["SuperTrend"][bear_st],
            name="SuperTrend (Ayı çizgi)", mode="lines",
            line=dict(color="rgba(255,60,60,0.5)", width=1.5),
            visible="legendonly", showlegend=True), row=1, col=1)

        if st_buy_signal.any():
            fig.add_trace(go.Scatter(
                x=df.index[st_buy_signal],
                y=df["SuperTrend"][st_buy_signal],
                name="SuperTrend AL",
                mode="markers+text",
                marker=dict(symbol="square", color="#00c853", size=18, line=dict(color="#00c853", width=0)),
                text="AL",
                textfont=dict(color="white", size=8, family="Arial Black"),
                textposition="middle center",
            ), row=1, col=1)

        if st_sell_signal.any():
            fig.add_trace(go.Scatter(
                x=df.index[st_sell_signal],
                y=df["SuperTrend"][st_sell_signal],
                name="SuperTrend SAT",
                mode="markers+text",
                marker=dict(symbol="square", color="#d50000", size=18, line=dict(color="#d50000", width=0)),
                text="SAT",
                textfont=dict(color="white", size=8, family="Arial Black"),
                textposition="middle center",
            ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Mid"],
            name="LRC Orta", visible="legendonly",
            line=dict(color="white", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Upper"],
            name="LRC Üst", visible="legendonly",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["LRC_Lower"],
            name="LRC Alt", visible="legendonly",
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.05)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["NW_Line"],
            name="NW Orta", visible="legendonly",
            line=dict(color="gold", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["NW_Upper"],
            name="NW Üst", visible="legendonly",
            line=dict(color="rgba(255,215,0,0.4)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["NW_Lower"],
            name="NW Alt", visible="legendonly",
            line=dict(color="rgba(255,215,0,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(255,215,0,0.04)"), row=1, col=1)

        if is_intraday:
            fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"],
                name="VWAP", visible="legendonly",
                line=dict(color="yellow", dash="dash", width=1.5)), row=1, col=1)

        FIB_COLORS = {
            "0.0%":   "rgba(128,128,128,0.7)",
            "23.6%":  "rgba(255,165,0,0.8)",
            "38.2%":  "rgba(255,215,0,0.9)",
            "50.0%":  "rgba(255,255,255,0.9)",
            "61.8%":  "rgba(255,215,0,0.9)",
            "78.6%":  "rgba(255,165,0,0.8)",
            "100.0%": "rgba(128,128,128,0.7)",
        }
        for lvl_name, lvl_price in fib_levels.items():
            fig.add_hline(
                y=lvl_price,
                line_dash="dot",
                line_color=FIB_COLORS.get(lvl_name, "gray"),
                line_width=1,
                annotation_text=f"  Fib {lvl_name} {lvl_price:.2f}",
                annotation_font=dict(color=FIB_COLORS.get(lvl_name, "gray"), size=9, family="monospace"),
                annotation_position="top left",
                row=1, col=1,
            )

        # ── Yatay S/R çizgileri (arka plan referans) ──────────────
        for lvl in swing_levels:
            is_support = lvl["type"] == "S"
            color      = "rgba(0,255,100,0.35)" if is_support else "rgba(255,80,80,0.35)"
            fig.add_hline(
                y=lvl["price"],
                line_color=color,
                line_width=1,
                line_dash="dot",
                row=1, col=1,
            )

        # ── Diyagonal Trend Çizgileri (legend toggle destekli) ────
        for tl in trendlines:
            is_sup  = tl["type"] == "support"
            color   = "rgba(0,255,120,0.9)" if is_sup else "rgba(255,80,80,0.9)"
            width   = 1 if tl["touches"] <= 2 else (2 if tl["touches"] <= 4 else 3)
            label   = f"{'↗ Destek' if is_sup else '↘ Direnç'} TL (x{tl['touches']})"
            x0_date = tl_dates[tl["x0"]]
            x1_date = tl_dates[tl["x1"]]
            fig.add_trace(go.Scatter(
                x=[x0_date, x1_date],
                y=[tl["y0"], tl["y1"]],
                mode="lines",
                name=label,
                line=dict(color=color, width=width, dash="solid"),
                visible="legendonly",
                legendgroup="trendlines",
                legendgrouptitle_text="Trend Çizgileri" if tl == trendlines[0] else None,
            ), row=1, col=1)

        # ── Kanal dolgusu (legend toggle destekli) ────────────────
        if tl_show_channel:
            for ci, ch in enumerate(tl_channels):
                sl   = ch["support"];  rl = ch["resistance"]
                xi0  = max(sl["x0"], rl["x0"])
                xi1  = sl["x1"]
                xs   = [tl_dates[xi0], tl_dates[xi1],
                        tl_dates[xi1], tl_dates[xi0], tl_dates[xi0]]
                y_s0 = sl["slope"] * xi0 + sl["intercept"]
                y_s1 = sl["slope"] * xi1 + sl["intercept"]
                y_r0 = rl["slope"] * xi0 + rl["intercept"]
                y_r1 = rl["slope"] * xi1 + rl["intercept"]
                ys   = [y_s0, y_s1, y_r1, y_r0, y_s0]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    fill="toself",
                    fillcolor="rgba(100,180,255,0.07)",
                    line=dict(width=0),
                    mode="lines",
                    name=f"Kanal {ci+1}",
                    visible="legendonly",
                    legendgroup="trendlines",
                    showlegend=True,
                ), row=1, col=1)
        # ──────────────────────────────────────────────────────────

        fig.add_trace(go.Bar(
            x=vol_at_price, y=bin_centers,
            orientation="h",
            marker_color=bar_colors,
            name="Hacim Profili",
            showlegend=False,
            hovertemplate="Fiyat: %{y:.2f}<br>Hacim: %{x:,.0f}<extra></extra>",
        ), row=1, col=2)

        fig.add_hline(y=poc_price, line_dash="dash", line_color="orange",
            annotation_text=f"POC {poc_price:.2f}",
            annotation_font=dict(color="orange", size=10, family="monospace"),
            annotation_bgcolor="rgba(255,165,0,0.15)",
            annotation_position="top right", row=1, col=2)
        fig.add_hline(y=lp, line_dash="dot", line_color="lime" if lp >= pp else "red",
            annotation_text=f"  {lp:.2f}",
            annotation_font=dict(color="lime" if lp >= pp else "red", size=12, family="monospace"),
            annotation_bgcolor="rgba(0,255,0,0.12)" if lp >= pp else "rgba(255,0,0,0.12)",
            annotation_position="bottom right", row=1, col=2)

        fig.add_annotation(text=f"<b>{ticker}  {lp:,.4f}</b>",
            xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
            font=dict(size=13, color="#007a3d" if lp >= pp else "#cc2200", family="monospace"),
            align="left", bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(200,200,200,0.5)", borderwidth=1, borderpad=4)

        fig.update_layout(
            template="plotly_dark", height=580,
            dragmode="pan",
            xaxis=dict(rangeslider_visible=False),
            xaxis2=dict(showgrid=False, showticklabels=False),
            yaxis2=dict(showticklabels=False),
            legend=dict(
                orientation="v",
                x=-0.02, y=1,
                xanchor="right", yanchor="top",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=8),
                itemwidth=30,
                itemsizing="constant",
                tracegroupgap=4,
            ),
            margin=dict(l=110, r=10, t=30, b=30),
        )

        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # ============================================================
        # ALT GRAFİKLER
        # ============================================================
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "RSI", "MACD", "ADX", "OBV", "Stoch RSI", "Ichimoku", "SuperTrend",
            "KAMA & LRC", "Nadaraya-Watson", "WaveTrend", "Divergence"])

        with tab1:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                line=dict(color="rgba(0,200,100,0.9)", width=1.5),
                fill="tozeroy", fillcolor="rgba(0,200,100,0.15)"))
            f.add_trace(go.Scatter(x=df.index, y=df["RSI_MA"],
                name=f"RSI MA({rsi_ma_period})", line=dict(color="yellow", width=1.5, dash="dot")))
            f.add_hline(y=p_rsi["rsi_lower"], line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({p_rsi['rsi_lower']})")
            f.add_hline(y=p_rsi["rsi_upper"], line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({p_rsi['rsi_upper']})")
            f.add_hline(y=50, line_dash="dot", line_color="gray")
            bull_div_rsi = df["Div_RSI"] == 1
            bear_div_rsi = df["Div_RSI"] == -1
            if bull_div_rsi.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_rsi], y=df["RSI"][bull_div_rsi],
                    name="Bullish Div", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if bear_div_rsi.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_rsi], y=df["RSI"][bear_div_rsi],
                    name="Bearish Div", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 RSI Nasıl Okunur?"):
                st.markdown("""
**RSI (Relative Strength Index)** — 0–100 arasında salınan momentum göstergesidir.

| Bölge | Anlam |
|---|---|
| RSI < Aşırı Satım eşiği | 🟢 Aşırı satılmış → potansiyel AL sinyali |
| RSI > Aşırı Alım eşiği | 🔴 Aşırı alınmış → potansiyel SAT sinyali |
| RSI ~ 50 | ⚪ Nötr bölge |

- **RSI MA (sarı noktalı):** RSI'nın hareketli ortalaması. RSI bu çizgiyi yukarı keserse momentum güçleniyor demektir.
- **Bullish Divergence 🔺:** Fiyat düşük dip yaparken RSI yüksek dip yapıyor → güçlü dönüş sinyali.
- **Bearish Divergence 🔻:** Fiyat yüksek tepe yaparken RSI alçak tepe yapıyor → zayıflama uyarısı.
                """)

        with tab2:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="cyan")))
            f.add_trace(go.Scatter(x=df.index, y=df["MACD_S"], name="Sinyal", line=dict(color="orange")))
            hist = df["MACD"] - df["MACD_S"]
            f.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                marker_color=["lime" if v >= 0 else "red" for v in hist], opacity=0.5))
            bull_div_macd = df["Div_MACD"] == 1
            bear_div_macd = df["Div_MACD"] == -1
            if bull_div_macd.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_macd], y=df["MACD"][bull_div_macd],
                    name="Bullish Div", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if bear_div_macd.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_macd], y=df["MACD"][bear_div_macd],
                    name="Bearish Div", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 MACD Nasıl Okunur?"):
                st.markdown("""
**MACD (Moving Average Convergence Divergence)** — trend yönü ve momentumu ölçer.

| Unsur | Anlam |
|---|---|
| MACD > Sinyal çizgisi | 🟢 Yukarı momentum → AL eğilimi |
| MACD < Sinyal çizgisi | 🔴 Aşağı momentum → SAT eğilimi |
| Histogram yeşil & büyüyor | 🟢 Momentum güçleniyor |
| Histogram kırmızı & büyüyor | 🔴 Momentum zayıflıyor |

- **Sıfır çizgisi geçişi:** MACD sıfırı yukarı kesiyor = güçlü boğa sinyali; aşağı kesiyor = ayı sinyali.
- **Bullish Divergence 🔺:** Fiyat düşük dip, MACD yüksek dip → trend dönüş öncüsü.
- **Bearish Divergence 🔻:** Fiyat yüksek tepe, MACD alçak tepe → zirve uyarısı.
                """)

        with tab3:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["ADX"],      name="ADX", line=dict(color="yellow", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["PLUS_DI"],  name="+DI", line=dict(color="lime", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["MINUS_DI"], name="-DI", line=dict(color="red",  dash="dot")))
            f.add_hline(y=p_adx["adx_threshold"], line_dash="dash", line_color="white",
                annotation_text=f"Trend Eşiği ({p_adx['adx_threshold']})")
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 ADX Nasıl Okunur?"):
                st.markdown("""
**ADX (Average Directional Index)** — trendin gücünü ölçer (yön değil, sadece güç).

| ADX Değeri | Trend Gücü |
|---|---|
| < 20 | Zayıf / yatay piyasa |
| 20–25 | Trend oluşuyor |
| > 25 | Güçlü trend |
| > 40 | Çok güçlü trend |

- **+DI (yeşil):** Yukarı yönlü hareketin gücü.
- **-DI (kırmızı):** Aşağı yönlü hareketin gücü.
- **+DI > -DI ve ADX > eşik:** 🟢 Güçlü yükseliş trendi.
- **-DI > +DI ve ADX > eşik:** 🔴 Güçlü düşüş trendi.
- ADX düşükken verilen sinyaller güvenilmezdir.
                """)

        with tab4:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="dodgerblue")))
            f.add_trace(go.Scatter(x=df.index, y=obv_sma_short,
                name=f"OBV SMA {obv_short}", line=dict(color="orange", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=obv_sma_long,
                name=f"OBV SMA {obv_long}", line=dict(color="cyan", dash="dot")))
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 OBV Nasıl Okunur?"):
                st.markdown("""
**OBV (On-Balance Volume)** — hacim akışını kümülatif olarak izler; fiyat hareketini önceden haber verebilir.

| Durum | Anlam |
|---|---|
| OBV yükseliyor, fiyat yükseliyor | 🟢 Trend onaylanıyor |
| OBV yükseliyor, fiyat düşüyor | 🟢 Gizli birikim → potansiyel yukarı kırılım |
| OBV düşüyor, fiyat yükseliyor | 🔴 Dağıtım var → zayıflama uyarısı |
| OBV düşüyor, fiyat düşüyor | 🔴 Trend onaylanıyor |

- **Kısa SMA (turuncu) > Uzun SMA (cyan):** OBV momentumu pozitif → AL eğilimi.
- **Kısa SMA < Uzun SMA:** OBV momentumu negatif → SAT eğilimi.
- OBV'nin mutlak değeri değil, eğimi önemlidir.
                """)

        with tab5:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["StochRSI_K"], name="%K", line=dict(color="magenta")))
            f.add_trace(go.Scatter(x=df.index, y=df["StochRSI_D"], name="%D", line=dict(color="orange", dash="dot")))
            f.add_hline(y=stoch_lower, line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({stoch_lower})")
            f.add_hline(y=stoch_upper, line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({stoch_upper})")
            f.update_layout(**sub_layout())
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Stochastic RSI Nasıl Okunur?"):
                st.markdown("""
**Stochastic RSI** — RSI'ya uygulanan Stochastic göstergesidir. RSI'dan daha hassas ve hızlıdır.

| Bölge | Anlam |
|---|---|
| %K < Aşırı Satım eşiği | 🟢 Aşırı satılmış → AL bölgesi |
| %K > Aşırı Alım eşiği | 🔴 Aşırı alınmış → SAT bölgesi |

- **%K (mor):** Hızlı çizgi — anlık sinyal verir.
- **%D (turuncu noktalı):** %K'nın ortalaması — yavaş, daha güvenilir.
- **%K, %D'yi aşırı satım bölgesinde yukarı kesiyor:** 🟢 Güçlü AL sinyali.
- **%K, %D'yi aşırı alım bölgesinde aşağı kesiyor:** 🔴 Güçlü SAT sinyali.
- RSI aşırı bölgelerde değilken Stoch RSI sinyalleri daha az güvenilirdir.
                """)

        with tab6:
            f = go.Figure()
            f.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Fiyat"))
            f.add_trace(go.Scatter(x=df.index, y=df["Tenkan"], name="Tenkan-sen", line=dict(color="cyan",  width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["Kijun"],  name="Kijun-sen",  line=dict(color="red",   width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["Senkou_A"], name="Senkou A",
                line=dict(color="lime", width=0.5, dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["Senkou_B"], name="Senkou B",
                line=dict(color="red", width=0.5, dash="dot"),
                fill="tonexty", fillcolor="rgba(100,100,100,0.15)"))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Ichimoku Nasıl Okunur?"):
                st.markdown("""
**Ichimoku Kinko Hyo** — trend yönü, destek/direnç ve momentum'u tek grafikte gösterir.

| Unsur | Renk | Anlam |
|---|---|---|
| Tenkan-sen | Cyan | Kısa vadeli denge çizgisi (9 bar) |
| Kijun-sen | Kırmızı | Orta vadeli denge çizgisi (26 bar) |
| Senkou Span A | Yeşil | Bulutun üst sınırı |
| Senkou Span B | Kırmızı | Bulutun alt sınırı |

**Okuma Kuralları:**
- **Fiyat bulutun üstünde:** 🟢 Yükseliş trendi.
- **Fiyat bulutun altında:** 🔴 Düşüş trendi.
- **Fiyat bulut içinde:** ⚪ Konsolidasyon.
- **Tenkan > Kijun:** 🟢 Kısa vadeli momentum pozitif.
- **Yeşil bulut (Span A > Span B):** Boğa piyasası.
- **Kırmızı bulut (Span B > Span A):** Ayı piyasası.
                """)

        with tab7:
            f = go.Figure()
            f.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Fiyat"))
            f.add_trace(go.Scatter(x=df.index[bull_st], y=df["SuperTrend"][bull_st],
                name="SuperTrend (Boğa)", mode="lines", line=dict(color="lime", width=2)))
            f.add_trace(go.Scatter(x=df.index[bear_st], y=df["SuperTrend"][bear_st],
                name="SuperTrend (Ayı)", mode="lines", line=dict(color="red", width=2)))
            if st_buy_signal.any():
                f.add_trace(go.Scatter(
                    x=df.index[st_buy_signal], y=df["SuperTrend"][st_buy_signal],
                    name="AL", mode="markers+text",
                    marker=dict(symbol="square", color="#00c853", size=18, line=dict(width=0)),
                    text="AL",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    textposition="middle center"))
            if st_sell_signal.any():
                f.add_trace(go.Scatter(
                    x=df.index[st_sell_signal], y=df["SuperTrend"][st_sell_signal],
                    name="SAT", mode="markers+text",
                    marker=dict(symbol="square", color="#d50000", size=18, line=dict(width=0)),
                    text="SAT",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    textposition="middle center"))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 SuperTrend Nasıl Okunur?"):
                st.markdown("""
**SuperTrend** — ATR tabanlı dinamik destek/direnç çizgisidir.

| Durum | Anlam |
|---|---|
| Çizgi yeşil (fiyatın altında) | 🟢 Yükseliş trendi — uzun pozisyon |
| Çizgi kırmızı (fiyatın üstünde) | 🔴 Düşüş trendi — kısa pozisyon |
| 🟩 AL kutusu | ⚡ Ayıdan boğaya geçiş — trend dönüşü |
| 🟥 SAT kutusu | ⚡ Boğadan ayıya geçiş — trend dönüşü |

- **ATR Çarpanı (multiplier):** Yüksek değer → daha az sinyal, daha az gürültü.
- **En güçlü sinyal:** SuperTrend AL/SAT + ADX > eşik değeri kombinasyonu.
- Yatay piyasalarda yanlış sinyal üretebilir; ADX filtresiyle kullanın.
                """)

        with tab8:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat", line=dict(color="white", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["KAMA"], name="KAMA", line=dict(color="violet", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Mid"], name="LRC Orta",
                line=dict(color="white", dash="dash", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Upper"], name="LRC Üst",
                line=dict(color="rgba(200,200,200,0.6)", dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["LRC_Lower"], name="LRC Alt",
                line=dict(color="rgba(200,200,200,0.6)", dash="dot"),
                fill="tonexty", fillcolor="rgba(150,150,150,0.07)"))
            f.update_layout(**sub_layout(height=350))
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 KAMA & LR Channel Nasıl Okunur?"):
                st.markdown("""
**KAMA (Kaufman Adaptive Moving Average)** — piyasa koşullarına göre hız adapte eden akıllı bir ortalamadır.

| Durum | Anlam |
|---|---|
| Fiyat > KAMA | 🟢 Yükseliş eğilimi |
| Fiyat < KAMA | 🔴 Düşüş eğilimi |
| KAMA düz seyrediyor | ⚪ Piyasa yatay, bekle |

**LR Channel (Linear Regression Channel)**

| Durum | Anlam |
|---|---|
| Fiyat alt banda değiyor | 🟢 Potansiyel destek / AL bölgesi |
| Fiyat üst banda değiyor | 🔴 Potansiyel direnç / SAT bölgesi |
                """)

        with tab9:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat", line=dict(color="white", width=1)))
            f.add_trace(go.Scatter(x=df.index, y=df["NW_Line"], name="NW Orta", line=dict(color="gold", width=2)))
            f.add_trace(go.Scatter(x=df.index, y=df["NW_Upper"], name="NW Üst",
                line=dict(color="red", width=1, dash="dot")))
            f.add_trace(go.Scatter(x=df.index, y=df["NW_Lower"], name="NW Alt",
                line=dict(color="lime", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(255,215,0,0.05)"))
            nw_ob = close > df["NW_Upper"]
            nw_os = close < df["NW_Lower"]
            if nw_ob.any():
                f.add_trace(go.Scatter(x=df.index[nw_ob], y=close[nw_ob],
                    name="Aşırı Alım", mode="markers", marker=dict(color="red", size=6)))
            if nw_os.any():
                f.add_trace(go.Scatter(x=df.index[nw_os], y=close[nw_os],
                    name="Aşırı Satım", mode="markers", marker=dict(color="lime", size=6)))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Nadaraya-Watson Nasıl Okunur?"):
                st.markdown("""
**Nadaraya-Watson Envelope** — çekirdek regresyon ile hesaplanan non-parametrik bir zarf göstergesidir.

| Durum | Anlam |
|---|---|
| Fiyat üst zarfın üstünde 🔴 | Aşırı alım — geri çekilme beklenebilir |
| Fiyat alt zarfın altında 🟢 | Aşırı satım — toparlanma beklenebilir |
| Fiyat zarf içinde | Normal seyir |
                """)

        with tab10:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=df["WT1"], name="WT1",
                line=dict(color="cyan", width=1.5)))
            f.add_trace(go.Scatter(x=df.index, y=df["WT2"], name="WT2",
                line=dict(color="orange", width=1.5, dash="dot")))
            wt_hist = df["WT1"] - df["WT2"]
            f.add_trace(go.Bar(x=df.index, y=wt_hist, name="WT Histogram",
                marker_color=["lime" if v >= 0 else "red" for v in wt_hist], opacity=0.4))
            f.add_hline(y=wt_ob, line_dash="dash", line_color="red",
                annotation_text=f"Aşırı Alım ({wt_ob})")
            f.add_hline(y=wt_os, line_dash="dash", line_color="lime",
                annotation_text=f"Aşırı Satım ({wt_os})")
            f.add_hline(y=0, line_dash="dot", line_color="gray")
            wt_buy  = df["Sig_WaveTrend"] == 1
            wt_sell = df["Sig_WaveTrend"] == -1
            if wt_buy.any():
                f.add_trace(go.Scatter(x=df.index[wt_buy], y=df["WT1"][wt_buy],
                    name="AL", mode="markers",
                    marker=dict(color="lime", size=10, symbol="triangle-up")))
            if wt_sell.any():
                f.add_trace(go.Scatter(x=df.index[wt_sell], y=df["WT1"][wt_sell],
                    name="SAT", mode="markers",
                    marker=dict(color="red", size=10, symbol="triangle-down")))
            f.update_layout(**sub_layout(height=300))
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 WaveTrend Nasıl Okunur?"):
                st.markdown("""
**WaveTrend (WT_CROSS_LB)** — momentum ve aşırı bölge tespiti için kullanılan osilatördür.

| Unsur | Anlam |
|---|---|
| WT1 (cyan) | Hızlı sinyal çizgisi |
| WT2 (turuncu noktalı) | Yavaş sinyal çizgisi |

- **WT1, WT2'yi aşırı satım bölgesinde yukarı kesiyor 🔺:** Güçlü AL sinyali.
- **WT1, WT2'yi aşırı alım bölgesinde aşağı kesiyor 🔻:** Güçlü SAT sinyali.
                """)

        with tab11:
            f = go.Figure()
            f.add_trace(go.Scatter(x=df.index, y=close, name="Fiyat",
                line=dict(color="red", width=1.5)))
            bull_div_r = df["Div_RSI"]  == 1
            bear_div_r = df["Div_RSI"]  == -1
            bull_div_m = df["Div_MACD"] == 1
            bear_div_m = df["Div_MACD"] == -1
            if bull_div_r.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_r], y=close[bull_div_r],
                    name="RSI Bullish Div", mode="markers",
                    marker=dict(color="lime", size=12, symbol="triangle-up")))
            if bear_div_r.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_r], y=close[bear_div_r],
                    name="RSI Bearish Div", mode="markers",
                    marker=dict(color="red", size=12, symbol="triangle-down")))
            if bull_div_m.any():
                f.add_trace(go.Scatter(x=df.index[bull_div_m], y=close[bull_div_m],
                    name="MACD Bullish Div", mode="markers",
                    marker=dict(color="aquamarine", size=10, symbol="diamond")))
            if bear_div_m.any():
                f.add_trace(go.Scatter(x=df.index[bear_div_m], y=close[bear_div_m],
                    name="MACD Bearish Div", mode="markers",
                    marker=dict(color="salmon", size=10, symbol="diamond")))
            f.update_layout(**sub_layout(height=350), xaxis_rangeslider_visible=False,
                title_text="Divergence Noktaları (Fiyat Grafiği Üzerinde)")
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
            with st.expander("📖 Divergence Nasıl Okunur?"):
                st.markdown("""
**Divergence (Uyumsuzluk)** — fiyat hareketi ile indikatör arasındaki zıtlık; trend dönüşünün erken habercisidir.

| Tür | Fiyat | İndikatör | Anlam |
|---|---|---|---|
| Bullish Div 🔺 | Düşük dip | Yüksek dip | 🟢 Satış baskısı azalıyor → yukarı dönüş olabilir |
| Bearish Div 🔻 | Yüksek tepe | Düşük tepe | 🔴 Alış gücü zayıflıyor → aşağı dönüş olabilir |
                """)

        # ============================================================
        # KARAR TABLOSU
        # ============================================================
        last       = df.iloc[-1]
        last_close = safe_scalar(last["Close"])
        last_ath   = bool(last["ATR_High"]) if not pd.isna(last["ATR_High"]) else False
        res        = []

        def trend_dec(raw_dec, atr_ok):
            return raw_dec if atr_ok else "TUT (düşük vol.)"

        lss = safe_scalar(last["SMA_SHORT"])
        lsl = safe_scalar(last["SMA_LONG"])
        if not (np.isnan(lss) or np.isnan(lsl)):
            res.append([trend_dec("AL" if lss > lsl else "SAT", last_ath),
                        f"SMA ({p_sma['sma_s']}/{p_sma['sma_l']})", "Trend yönü."])
        else:
            res.append(["N/A", "SMA Crossover", "Yetersiz veri."])

        lr = safe_scalar(last["RSI"])
        if not np.isnan(lr):
            dec = "AL" if lr < p_rsi["rsi_lower"] else ("SAT" if lr > p_rsi["rsi_upper"] else "TUT")
            res.append([dec, f"RSI ({p_rsi['rsi_period']}) [{p_rsi['rsi_lower']}/{p_rsi['rsi_upper']}]", f"Seviye: {lr:.1f}"])
        else:
            res.append(["N/A", "RSI", "Yetersiz veri."])

        lup = safe_scalar(last["Up"])
        llb = safe_scalar(last["Low_BB"])
        if not any(np.isnan(v) for v in [last_close, llb, lup]):
            dec = "AL" if last_close < llb else ("SAT" if last_close > lup else "TUT")
            res.append([dec, f"Bollinger Bands (σ={p_bb['bb_std']})", "Fiyatın kanaldaki yeri."])
        else:
            res.append(["N/A", "Bollinger Bands", "Yetersiz veri."])

        lm  = safe_scalar(last["MACD"])
        lms = safe_scalar(last["MACD_S"])
        if not (np.isnan(lm) or np.isnan(lms)):
            res.append([trend_dec("AL" if lm > lms else "SAT", last_ath),
                        f"MACD ({p_macd['macd_fast']},{p_macd['macd_slow']},{macd_signal})", "Momentum."])
        else:
            res.append(["N/A", "MACD", "Yetersiz veri."])

        lz = safe_scalar(last["Z"])
        if not np.isnan(lz):
            dec = "AL" if lz < -p_z["z_thresh"] else ("SAT" if lz > p_z["z_thresh"] else "TUT")
            res.append([dec, f"Mean Reversion (z={p_z['z_thresh']})", f"Z: {lz:.2f}"])
        else:
            res.append(["N/A", "Mean Reversion", "Yetersiz veri."])

        lo = safe_scalar(last["Sig_OBV"])
        if lo != 0 and not np.isnan(lo):
            res.append(["AL" if lo > 0 else "SAT", f"OBV ({obv_short}/{obv_long})", "Hacim trendi."])
        else:
            res.append(["N/A", f"OBV ({obv_short}/{obv_long})", "Yetersiz veri."])

        la   = safe_scalar(last["ADX"])
        lpd  = safe_scalar(last["PLUS_DI"])
        lmd2 = safe_scalar(last["MINUS_DI"])
        if not np.isnan(la):
            if la > p_adx["adx_threshold"]:
                res.append([trend_dec("AL" if lpd > lmd2 else "SAT", last_ath),
                             "ADX", f"ADX: {la:.1f} (Güçlü, eşik={p_adx['adx_threshold']})"])
            else:
                res.append(["TUT", "ADX", f"ADX: {la:.1f} (Zayıf, eşik={p_adx['adx_threshold']})"])
        else:
            res.append(["N/A", "ADX", "Yetersiz veri."])

        if is_intraday:
            lv  = safe_scalar(last["VWAP"])
            lvs = safe_scalar(last["Sig_VWAP"])
            if not np.isnan(lv):
                dec = "AL" if lvs == 1 else ("SAT" if lvs == -1 else "TUT")
                res.append([dec, "VWAP", f"VWAP: {lv:.2f} | bant: ±%{vwap_band_pct:.2f}"])
            else:
                res.append(["N/A", "VWAP", "Yetersiz veri."])
        else:
            res.append(["N/A", "VWAP", "Günlük+ periyotta devre dışı."])

        lsk = float(df["StochRSI_K"].iloc[-1])
        if not np.isnan(lsk):
            dec = "AL" if lsk < stoch_lower else ("SAT" if lsk > stoch_upper else "TUT")
            res.append([dec, f"Stoch RSI ({stoch_rsi_period})", f"%K: {lsk:.1f}"])
        else:
            res.append(["N/A", "Stoch RSI", "Yetersiz veri."])

        lis = safe_scalar(last["Sig_Ichimoku"])
        if not last_ath and lis != 0:
            res.append(["TUT (düşük vol.)", "Ichimoku", "ATR filtresi aktif."])
        elif lis == 1:  res.append(["AL",  "Ichimoku", "Tenkan > Kijun, fiyat bulut üstünde."])
        elif lis == -1: res.append(["SAT", "Ichimoku", "Tenkan < Kijun, fiyat bulut altında."])
        else:           res.append(["TUT", "Ichimoku", "Karışık sinyal / bulut içinde."])

        lk = safe_scalar(last["KAMA"])
        if not np.isnan(lk):
            res.append([trend_dec("AL" if last_close > lk else "SAT", last_ath),
                        f"KAMA ({kama_period},{kama_fast},{kama_slow})", f"KAMA: {lk:.2f}"])
        else:
            res.append(["N/A", "KAMA", "Yetersiz veri."])

        lst  = safe_scalar(last["SuperTrend"])
        lstd = safe_scalar(last["ST_Direction"])
        if not np.isnan(lst):
            res.append([trend_dec("AL" if lstd == 1 else "SAT", last_ath),
                        f"SuperTrend ({p_st['st_period']}, x{p_st['st_multiplier']})", f"Seviye: {lst:.2f}"])
        else:
            res.append(["N/A", "SuperTrend", "Yetersiz veri."])

        llrc = safe_scalar(last["Sig_LRC"])
        llm  = safe_scalar(last["LRC_Mid"])
        if not np.isnan(llm):
            dec = "AL" if llrc == 1 else ("SAT" if llrc == -1 else "TUT")
            res.append([dec, f"LR Channel (σ={p_lrc['lrc_std_mult']})", f"Orta: {llm:.2f}"])
        else:
            res.append(["N/A", "LR Channel", "Yetersiz veri."])

        la2 = safe_scalar(last["ATR"])
        lam = safe_scalar(atr_ma.iloc[-1])
        if not np.isnan(la2):
            res.append(["BİLGİ", "ATR Filtre",
                f"Volatilite: {'Yüksek ↑' if last_ath else 'Düşük ↓'} | ATR: {la2:.2f} | MA: {lam:.2f}"])
        else:
            res.append(["N/A", "ATR Filtre", "Yetersiz veri."])

        lnw = safe_scalar(last["NW_Line"])
        lnu = safe_scalar(last["NW_Upper"])
        lnl = safe_scalar(last["NW_Lower"])
        if not np.isnan(lnw):
            if last_close > lnu:   nw_note = "Üst zarfın üstünde (aşırı alım)"
            elif last_close < lnl: nw_note = "Alt zarfın altında (aşırı satım)"
            else:                  nw_note = f"Zarf içinde. NW: {lnw:.2f}"
            res.append(["BİLGİ", "Nadaraya-Watson", nw_note])
        else:
            res.append(["N/A", "Nadaraya-Watson", "Yetersiz veri."])

        lwt1    = safe_scalar(last["WT1"])
        lwt_sig = safe_scalar(last["Sig_WaveTrend"])
        if not np.isnan(lwt1):
            if lwt1 > wt_ob:   wt_zone = f"Aşırı Alım (WT1={lwt1:.1f})"
            elif lwt1 < wt_os: wt_zone = f"Aşırı Satım (WT1={lwt1:.1f})"
            else:               wt_zone = f"Nötr Bölge (WT1={lwt1:.1f})"
            wt_dec = "AL" if lwt_sig == 1 else ("SAT" if lwt_sig == -1 else "TUT")
            res.append([wt_dec, f"WaveTrend ({p_wt['wt_n1']}/{p_wt['wt_n2']})", wt_zone])
        else:
            res.append(["N/A", "WaveTrend", "Yetersiz veri."])

        # ── YENİ: EMA200 karar satırı ─────────────────────────────
        lema200 = safe_scalar(last["EMA200"])
        if not np.isnan(lema200):
            ema_dec = trend_dec("AL" if last_close > lema200 else "SAT", last_ath)
            res.append([ema_dec, "EMA 200", f"EMA200: {lema200:.2f} | Fiyat {'üstünde ✅' if last_close > lema200 else 'altında ❌'}"])
        else:
            res.append(["N/A", "EMA 200", "Yetersiz veri (min 200 bar gerekli)."])

        # ── YENİ: En yakın S/R seviyesi karar satırı ──────────────
        if swing_levels:
            closest_sr = min(swing_levels, key=lambda x: abs(x["price"] - last_close))
            dist_pct   = abs(closest_sr["price"] - last_close) / last_close * 100
            sr_label   = "Destek" if closest_sr["type"] == "S" else "Direnç"
            res.append(["BİLGİ", "Swing S/R",
                f"En yakın {sr_label}: {closest_sr['price']:.2f} "
                f"(%{dist_pct:.1f} uzakta, {closest_sr['touches']}x dokunuş)"])
        # ──────────────────────────────────────────────────────────

        last_div_rsi  = safe_scalar(last["Div_RSI"])
        last_div_macd = safe_scalar(last["Div_MACD"])
        if last_div_rsi == 1:
            res.append(["BİLGİ", "Divergence (RSI)", "🔺 Bullish Divergence — güçlü dip sinyali olabilir"])
        elif last_div_rsi == -1:
            res.append(["BİLGİ", "Divergence (RSI)", "🔻 Bearish Divergence — zayıflayan momentum"])
        else:
            res.append(["BİLGİ", "Divergence (RSI)", "Aktif divergence yok"])
        if last_div_macd == 1:
            res.append(["BİLGİ", "Divergence (MACD)", "🔺 Bullish Divergence"])
        elif last_div_macd == -1:
            res.append(["BİLGİ", "Divergence (MACD)", "🔻 Bearish Divergence"])
        else:
            res.append(["BİLGİ", "Divergence (MACD)", "Aktif divergence yok"])

        if fib_levels:
            closest_lvl = min(fib_levels.items(), key=lambda x: abs(x[1] - last_close))
            res.append(["BİLGİ", f"Fibonacci ({fib_lookback} bar)",
                        f"En yakın seviye: {closest_lvl[0]} ({closest_lvl[1]:.2f}) | Swing: {fib_low:.2f} — {fib_high:.2f}"])

        valid_sigs = [x for x in res if x[0] in ("AL", "SAT")]
        al_count   = sum(1 for x in valid_sigs if x[0] == "AL")
        sat_count  = sum(1 for x in valid_sigs if x[0] == "SAT")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anlık Fiyat",  f"{last_close:.2f}")
        c2.metric("AL Sinyali",   f"{al_count}")
        c3.metric("SAT Sinyali",  f"{sat_count}")
        c4.metric("Zaman Dilimi", f"{interval}")

        st.subheader("🔍 Algoritmik Detaylar")
        res_df = pd.DataFrame(res, columns=["Karar", "Algoritma", "Durum/Sebep"])

        def color_map(val):
            if val == "AL":    return "color: #00ff00; font-weight: bold"
            if val == "SAT":   return "color: #ff4b4b; font-weight: bold"
            if val == "N/A":   return "color: #ffaa00; font-weight: bold"
            if val == "BİLGİ": return "color: #00bfff; font-weight: bold"
            if "düşük vol." in str(val): return "color: #808495; font-style: italic"
            return "color: #808495; font-weight: bold"

        st.table(res_df.style.map(color_map, subset=["Karar"]))

        # ============================================================
        # BİREYSEL BACKTEST TABLOSU
        # ============================================================
        st.write("---")
        st.header("📊 Bireysel Algoritma Backtesti (Güncel Parametrelerle)")
        st.caption("⚠️ Geçmiş performans gelecekteki sonuçların garantisi değildir.")

        algo_signal_map = {
            f"SMA ({p_sma['sma_s']}/{p_sma['sma_l']})":                                    "Sig_SMA",
            f"RSI (p={p_rsi['rsi_period']} [{p_rsi['rsi_lower']}/{p_rsi['rsi_upper']}])":  "Sig_RSI",
            f"Bollinger Bands (p={p_bb['bb_period']}, σ={p_bb['bb_std']})":                "Sig_BB",
            f"MACD ({p_macd['macd_fast']},{p_macd['macd_slow']},{p_macd['macd_signal']})": "Sig_MACD",
            f"Mean Reversion (p={p_z['z_period']}, z={p_z['z_thresh']})":                  "Sig_Z",
            "OBV":                                                                           "Sig_OBV",
            f"ADX (p={p_adx['adx_period']}, eşik={p_adx['adx_threshold']})":              "Sig_ADX",
            "Stoch RSI":                                                                     "Sig_StochRSI",
            "Ichimoku":                                                                      "Sig_Ichimoku",
            "KAMA":                                                                          "Sig_KAMA",
            f"SuperTrend (p={p_st['st_period']}, x{p_st['st_multiplier']})":               "Sig_SuperTrend",
            f"LR Channel (p={p_lrc['lrc_period']}, σ={p_lrc['lrc_std_mult']})":           "Sig_LRC",
            f"WaveTrend ({p_wt['wt_n1']}/{p_wt['wt_n2']})":                               "Sig_WaveTrend",
        }
        if is_intraday:
            algo_signal_map["VWAP"] = "Sig_VWAP"

        algo_results = []
        for algo_name, sig_col in algo_signal_map.items():
            if sig_col not in df.columns:
                continue
            stats = run_backtest(df[sig_col], close_arr, cost_pct)
            algo_results.append({
                "Algoritma":       algo_name,
                "Trade":           stats["n"],
                "Getiri (%)":      round(stats["total_ret"], 2),
                "Win Rate (%)":    round(stats["win_rate"],  1),
                "Ort. Kazanç (%)": round(stats["avg_win"],  2),
                "Ort. Kayıp (%)":  round(stats["avg_loss"], 2),
                "Max DD (%)":      round(stats["max_dd"],   2),
                "Profit Factor":   round(stats["pf"], 2) if stats["pf"] != float("inf") else "∞",
            })

        if algo_results:
            algo_df = pd.DataFrame(algo_results)
            active  = algo_df[algo_df["Trade"] > 0].copy()
            if not active.empty:
                best = active.loc[active["Getiri (%)"].idxmax()]
                st.success(
                    f"🥇 En iyi: **{best['Algoritma']}** — "
                    f"Getiri: %{best['Getiri (%)']}, "
                    f"Win Rate: %{best['Win Rate (%)']}, "
                    f"{int(best['Trade'])} trade")
            def ret_color(val):
                if isinstance(val, (int, float)):
                    if val > 0: return "color: #00ff00"
                    if val < 0: return "color: #ff4b4b"
                return ""
            st.dataframe(algo_df.style.map(ret_color, subset=["Getiri (%)"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("Algoritma performansı hesaplanamadı.")

        # ============================================================
        # OPTİMİZASYON ÖZET TABLOSU
        # ============================================================
        st.write("---")
        st.subheader("🧬 Walk-Forward Optimizasyon Sonuçları")
        st.caption(f"{n_windows} pencere · %{train_pct} eğitim / %{100-train_pct} test · kriter: Sharpe")

        def opt_color(val):
            if isinstance(val, (int, float)):
                if val > 0: return "color: #00ff00"
                if val < 0: return "color: #ff4b4b"
            return ""

        score_col = "Ort. Test Sharpe"
        opt_rows  = []
        for algo_name, grid in PARAM_GRIDS.items():
            p = opt_params.get(algo_name, {})
            s = opt_stats.get(algo_name, {})
            row = {"Algoritma": algo_name}
            param_str            = "  |  ".join(f"{k} = {v}" for k, v in p.items())
            row["Parametreler"]  = param_str
            row["Getiri (%)"]    = round(s.get("total_ret", 0), 2)
            row["Sharpe"]        = round(s.get("sharpe",    0), 2)
            row["Trade"]         = s.get("n", 0)
            row["Win Rate (%)"]  = round(s.get("win_rate",  0), 1)
            row[score_col]       = round(s.get("wf_avg_score", 0), 3)
            opt_rows.append(row)

        opt_df     = pd.DataFrame(opt_rows)
        color_cols = [c for c in ["Getiri (%)", "Sharpe", score_col] if c in opt_df.columns]
        fmt        = {"Getiri (%)": "{:.2f}", "Sharpe": "{:.2f}", "Win Rate (%)": "{:.1f}", score_col: "{:.3f}"}
        fmt        = {k: v for k, v in fmt.items() if k in opt_df.columns}
        st.dataframe(
            opt_df.style.format(fmt).map(opt_color, subset=color_cols),
            use_container_width=True, hide_index=True)

        # ============================================================
        # RAPOR BÖLÜMÜ
        # ============================================================
        st.write("---")
        st.header("📋 Teknik Analiz Raporu")
        st.caption("Kural tabanlı otomatik analiz. Yatırım tavsiyesi içermez.")

        r_close    = safe_scalar(last["Close"])
        r_kama     = safe_scalar(last["KAMA"])
        r_adx      = safe_scalar(last["ADX"])
        r_pdi      = safe_scalar(last["PLUS_DI"])
        r_mdi      = safe_scalar(last["MINUS_DI"])
        r_macd     = safe_scalar(last["MACD"])
        r_macds    = safe_scalar(last["MACD_S"])
        r_rsi      = safe_scalar(last["RSI"])
        r_stk      = safe_scalar(last["StochRSI_K"])
        r_std      = safe_scalar(last["ST_Direction"])
        r_lrc_sig  = safe_scalar(last["Sig_LRC"])
        r_lrc_mid  = safe_scalar(last["LRC_Mid"])
        r_lrc_up   = safe_scalar(last["LRC_Upper"])
        r_lrc_lo   = safe_scalar(last["LRC_Lower"])
        r_nw       = safe_scalar(last["NW_Line"])
        r_nw_up    = safe_scalar(last["NW_Upper"])
        r_nw_lo    = safe_scalar(last["NW_Lower"])
        r_vwap     = safe_scalar(last["VWAP"])     if is_intraday else np.nan
        r_vwap_sig = safe_scalar(last["Sig_VWAP"]) if is_intraday else 0
        r_obv_sig  = safe_scalar(last["Sig_OBV"])
        r_div_rsi  = safe_scalar(last["Div_RSI"])
        r_div_mac  = safe_scalar(last["Div_MACD"])
        r_ichi     = safe_scalar(last["Sig_Ichimoku"])
        r_wt1      = safe_scalar(last["WT1"])
        r_atr_hi   = bool(last["ATR_High"]) if not pd.isna(last["ATR_High"]) else False
        r_ema200   = safe_scalar(last["EMA200"])

        if fib_levels:
            r_fib_closest = min(fib_levels.items(), key=lambda x: abs(x[1] - r_close))
        else:
            r_fib_closest = ("N/A", r_close)

        steps = []

        # ADIM 1 — Büyük Resim (EMA200 eklendi)
        kama_pos  = "ÜSTÜNDE ✅" if r_close > r_kama else "ALTINDA ❌"
        st_signal = "AL ✅" if r_std == 1 else "SAT ❌"
        poc_dist  = abs(r_close - poc_price) / r_close * 100
        ema200_pos = "üstünde ✅" if r_close > r_ema200 else "altında ❌"
        steps.append({
            "Adım": "1 — Büyük Resim",
            "Gösterge": "Fiyat / KAMA / EMA200 / SuperTrend / POC",
            "Değer": f"Fiyat: {r_close:.2f} | KAMA: {r_kama:.2f} | EMA200: {r_ema200:.2f} | POC: {poc_price:.2f} (%{poc_dist:.1f} uzakta)",
            "Yorum": f"Fiyat KAMA'nın {kama_pos} | EMA200 {ema200_pos} | SuperTrend: {st_signal} | En yakın Fib: {r_fib_closest[0]} ({r_fib_closest[1]:.2f})",
            "Sinyal": "AL" if (r_close > r_kama and r_std == 1 and r_close > r_ema200)
                      else ("SAT" if (r_close < r_kama and r_std == -1 and r_close < r_ema200) else "NÖTR"),
        })

        # ADIM 2 — Trend Gücü
        if not np.isnan(r_adx):
            adx_str  = f"ADX: {r_adx:.1f}"
            adx_sig  = "GÜÇLÜ TREND ✅" if r_adx > adx_threshold else "ZAYIF / YATAY ⚠️"
            adx_hint = "Trend sinyallerine güven" if r_adx > adx_threshold else "Mean-reversion sinyallerine ağırlık ver"
            steps.append({
                "Adım": "2 — Trend Gücü",
                "Gösterge": "ADX",
                "Değer": f"{adx_str} (eşik: {adx_threshold})",
                "Yorum": f"{adx_sig} — {adx_hint}",
                "Sinyal": "GÜÇLÜ" if r_adx > adx_threshold else "ZAYIF",
            })
        else:
            steps.append({"Adım": "2 — Trend Gücü", "Gösterge": "ADX",
                          "Değer": "N/A", "Yorum": "Yetersiz veri", "Sinyal": "N/A"})

        # ADIM 3 — Trend Yönü
        macd_pos  = r_macd > r_macds
        ichi_bull = r_ichi == 1
        ichi_bear = r_ichi == -1
        macd_str  = "Pozitif ✅" if macd_pos else "Negatif ❌"
        ichi_str  = "Boğa ✅" if ichi_bull else ("Ayı ❌" if ichi_bear else "Nötr ⚠️")
        trend_sig = "AL" if (macd_pos and ichi_bull) else ("SAT" if (not macd_pos and ichi_bear) else "NÖTR")
        steps.append({
            "Adım": "3 — Trend Yönü",
            "Gösterge": "MACD + Ichimoku",
            "Değer": f"MACD: {r_macd:.4f} | Sinyal: {r_macds:.4f}",
            "Yorum": f"MACD: {macd_str} | Ichimoku: {ichi_str}",
            "Sinyal": trend_sig,
        })

        # ADIM 4 — Giriş Noktası
        rsi_os    = r_rsi < rsi_lower
        rsi_ob    = r_rsi > rsi_upper
        stk_os    = r_stk < stoch_lower
        stk_ob    = r_stk > stoch_upper
        rsi_str   = f"Aşırı Satım ✅ ({r_rsi:.1f})" if rsi_os else (f"Aşırı Alım ❌ ({r_rsi:.1f})" if rsi_ob else f"Nötr ({r_rsi:.1f})")
        stk_str   = f"Aşırı Satım ✅ ({r_stk:.1f})" if stk_os else (f"Aşırı Alım ❌ ({r_stk:.1f})" if stk_ob else f"Nötr ({r_stk:.1f})")
        entry_sig = "AL" if (rsi_os or stk_os) else ("SAT" if (rsi_ob or stk_ob) else "BEKLE")
        steps.append({
            "Adım": "4 — Giriş Noktası",
            "Gösterge": "RSI + Stoch RSI",
            "Değer": f"RSI: {r_rsi:.1f} | Stoch %K: {r_stk:.1f}",
            "Yorum": f"RSI: {rsi_str} | Stoch RSI: {stk_str}",
            "Sinyal": entry_sig,
        })

        # ADIM 5 — Seviye Teyidi (S/R eklendi)
        lrc_str   = "Alt banda yakın ✅" if r_lrc_sig == 1 else ("Üst banda yakın ❌" if r_lrc_sig == -1 else "Kanal ortası ⚪")
        nw_str    = ("Üst zarfın üstünde ❌" if r_close > r_nw_up
                     else ("Alt zarfın altında ✅" if r_close < r_nw_lo else "Zarf içinde ⚪"))
        sr_note   = ""
        if swing_levels:
            csr      = min(swing_levels, key=lambda x: abs(x["price"] - r_close))
            sr_note  = f" | En yakın {'Destek ✅' if csr['type']=='S' else 'Direnç ❌'}: {csr['price']:.2f} ({csr['touches']}x)"
        level_sig = "AL" if (r_lrc_sig == 1 or r_close < r_nw_lo) else ("SAT" if (r_lrc_sig == -1 or r_close > r_nw_up) else "NÖTR")
        steps.append({
            "Adım": "5 — Seviye Teyidi",
            "Gösterge": "LR Channel + NW + S/R",
            "Değer": f"LRC Alt: {r_lrc_lo:.2f} | LRC Üst: {r_lrc_up:.2f} | NW: {r_nw:.2f}",
            "Yorum": f"LRC: {lrc_str} | NW: {nw_str}{sr_note}",
            "Sinyal": level_sig,
        })

        # ADIM 5b — VWAP
        if is_intraday and not np.isnan(r_vwap):
            vwap_pos = r_close > r_vwap
            vwap_str = f"Fiyat VWAP {'üstünde ✅' if vwap_pos else 'altında ❌'} | VWAP: {r_vwap:.2f}"
            vwap_dec = "AL" if r_vwap_sig == 1 else ("SAT" if r_vwap_sig == -1 else "NÖTR")
            steps.append({
                "Adım": "5b — VWAP Seviyesi",
                "Gösterge": "VWAP",
                "Değer": f"VWAP: {r_vwap:.2f} | Bant: ±%{vwap_band_pct:.2f}",
                "Yorum": vwap_str,
                "Sinyal": vwap_dec,
            })

        # ADIM 6 — Hacim Onayı
        obv_str = "Birikim ✅ (kısa SMA > uzun SMA)" if r_obv_sig == 1 else ("Dağıtım ❌ (kısa SMA < uzun SMA)" if r_obv_sig == -1 else "Nötr ⚪")
        steps.append({
            "Adım": "6 — Hacim Onayı",
            "Gösterge": "OBV",
            "Değer": f"Sinyal: {int(r_obv_sig)}",
            "Yorum": obv_str,
            "Sinyal": "AL" if r_obv_sig == 1 else ("SAT" if r_obv_sig == -1 else "NÖTR"),
        })

        # ADIM 7 — Uyarı / Divergence
        div_rsi_str  = ("🔺 Bullish Divergence — güçlü dip sinyali" if r_div_rsi == 1
                        else ("🔻 Bearish Divergence — zayıflama uyarısı" if r_div_rsi == -1 else "Yok ✅"))
        div_macd_str = ("🔺 Bullish Divergence" if r_div_mac == 1
                        else ("🔻 Bearish Divergence" if r_div_mac == -1 else "Yok ✅"))
        div_risk  = (r_div_rsi == -1 or r_div_mac == -1)
        div_boost = (r_div_rsi == 1  or r_div_mac == 1)
        steps.append({
            "Adım": "7 — Uyarı Kontrolü",
            "Gösterge": "Divergence (RSI + MACD)",
            "Değer": f"RSI Div: {int(r_div_rsi)} | MACD Div: {int(r_div_mac)}",
            "Yorum": f"RSI: {div_rsi_str} | MACD: {div_macd_str}",
            "Sinyal": "UYARI ❌" if div_risk else ("GÜÇLENDIRICI ✅" if div_boost else "TEMİZ ✅"),
        })

        report_df = pd.DataFrame(steps)

        def report_color(val):
            if "AL" in str(val) and "SAT" not in str(val): return "color: #00ff00; font-weight: bold"
            if "SAT" in str(val):    return "color: #ff4b4b; font-weight: bold"
            if "UYARI" in str(val):  return "color: #ff4b4b; font-weight: bold"
            if "TEMİZ" in str(val) or "GÜÇLEND" in str(val): return "color: #00bfff; font-weight: bold"
            if "NÖTR" in str(val) or "BEKLE" in str(val):    return "color: #aaaaaa"
            if "ZAYIF" in str(val):  return "color: #ffaa00; font-weight: bold"
            if "GÜÇLÜ" in str(val):  return "color: #00ff00; font-weight: bold"
            return ""

        st.dataframe(
            report_df.style.map(report_color, subset=["Sinyal"]),
            use_container_width=True, hide_index=True
        )

        al_onay  = sum(1 for s in steps if s["Sinyal"] in ("AL", "GÜÇLÜ", "GÜÇLENDIRICI ✅", "TEMİZ ✅"))
        sat_onay = sum(1 for s in steps if "SAT" in s["Sinyal"] or "UYARI" in s["Sinyal"])
        toplam   = len(steps)

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ AL Onayı",  f"{al_onay} / {toplam}")
        col2.metric("❌ SAT/Uyarı", f"{sat_onay} / {toplam}")
        col3.metric("Volatilite",   "Yüksek ↑" if r_atr_hi else "Düşük ↓")

        st.subheader("📝 Özet")
        ozet_parcalar = []

        if r_adx > adx_threshold:
            ozet_parcalar.append(f"ADX {r_adx:.1f} ile **güçlü bir trend** mevcut.")
        else:
            ozet_parcalar.append(f"ADX {r_adx:.1f} — piyasa **yatay seyirde**, mean-reversion sinyalleri daha geçerli.")

        if not np.isnan(r_ema200):
            if r_close > r_ema200:
                ozet_parcalar.append(f"Fiyat EMA200 ({r_ema200:.2f}) üstünde — **uzun vadeli trend pozitif**.")
            else:
                ozet_parcalar.append(f"Fiyat EMA200 ({r_ema200:.2f}) altında — **uzun vadeli trend negatif**.")

        if r_std == 1:
            ozet_parcalar.append("SuperTrend **AL** konumunda, yükseliş trendi destekleniyor.")
        else:
            ozet_parcalar.append("SuperTrend **SAT** konumunda, düşüş baskısı var.")

        if macd_pos:
            ozet_parcalar.append("MACD pozitif — momentum yukarı yönlü.")
        else:
            ozet_parcalar.append("MACD negatif — momentum aşağı yönlü.")

        if rsi_os:
            ozet_parcalar.append(f"RSI {r_rsi:.1f} ile **aşırı satım** bölgesinde — potansiyel giriş noktası.")
        elif rsi_ob:
            ozet_parcalar.append(f"RSI {r_rsi:.1f} ile **aşırı alım** bölgesinde — dikkatli olunmalı.")

        if swing_levels:
            csr = min(swing_levels, key=lambda x: abs(x["price"] - r_close))
            dist = abs(csr["price"] - r_close) / r_close * 100
            ozet_parcalar.append(
                f"En yakın {'destek' if csr['type']=='S' else 'direnç'}: "
                f"**{csr['price']:.2f}** (%{dist:.1f} uzakta, {csr['touches']}x test edilmiş).")

        if is_intraday and not np.isnan(r_vwap):
            if r_close > r_vwap:
                ozet_parcalar.append(f"Fiyat VWAP ({r_vwap:.2f}) üstünde — intraday momentum pozitif.")
            else:
                ozet_parcalar.append(f"Fiyat VWAP ({r_vwap:.2f}) altında — intraday baskı var.")

        if r_obv_sig == 1:
            ozet_parcalar.append("OBV birikim sinyali veriyor — hacim fiyatı destekliyor.")
        elif r_obv_sig == -1:
            ozet_parcalar.append("OBV dağıtım sinyali veriyor — hacim zayıflıyor.")

        if div_risk:
            ozet_parcalar.append("⚠️ **Bearish divergence** tespit edildi — mevcut sinyaller zayıflayabilir.")
        if div_boost:
            ozet_parcalar.append("🔺 **Bullish divergence** mevcut — AL sinyalini güçlendiriyor.")

        if al_onay >= 5 and not div_risk:
            ozet_parcalar.append(f"\n> 🟢 **Genel Değerlendirme: {al_onay}/{toplam} onay — güçlü AL sinyali.**")
        elif al_onay >= 3 and not div_risk:
            ozet_parcalar.append(f"\n> 🟡 **Genel Değerlendirme: {al_onay}/{toplam} onay — zayıf AL, teyit bekle.**")
        elif sat_onay >= 3 or div_risk:
            ozet_parcalar.append(f"\n> 🔴 **Genel Değerlendirme: {sat_onay} uyarı — SAT/bekle baskın.**")
        else:
            ozet_parcalar.append("\n> ⚪ **Genel Değerlendirme: Karışık sinyal — bekle.**")

        st.markdown(" ".join(ozet_parcalar))

    else:
        st.error("Veri çekilemedi. Ticker veya internet bağlantısını kontrol edin.")
