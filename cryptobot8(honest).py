#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cryptobot10_hft.py — cost-aware HFT trainer/evaluator (honest, fixed fwd returns)

Fixes & upgrades (vs. 9_hft):
- **BUGFIX**: edge & PnL now use **forward return (fwd1)** to match next-bar labeling.
- Feature **standardization** + higher max_iter to end LogisticRegression convergence spam.
- Clearer edge prints: μ+ / μ− in **bps**, threshold in **bps**.
- Same honest gate: trade only if E[r] > round-trip slippage.
- Same exits: time stop + micro TP/SL; vol-scaled fractional-Kelly sizing.
"""

import os, math, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, List

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)

# ------------------------------- Config -------------------------------

@dataclass
class CFG:
    csv_path: str = os.getenv("INPUT_CSV", "bonkers.csv")
    symbols: Tuple[str, ...] = ("BTC","ETH","DOGE")

    # minutes per bar
    timeframes_min: Tuple[int, ...] = (1, 3, 5, 10, 15)

    # walk-forward windows
    train_days: int = 3
    test_minutes: int = 30
    step_fraction: float = 1.0

    # features
    rsi_window: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    vwap_win: int = 20
    atr_win: int = 14
    vol_window: int = 30

    # costs (0 commish, normal slippage)
    slippage_bps_per_side: float = 2.0
    spread_bps: float = 0.0
    @property
    def slip_side(self) -> float:
        return (self.slippage_bps_per_side + self.spread_bps) / 10000.0
    @property
    def slip_roundtrip(self) -> float:
        return 2.0 * self.slip_side

    # position sizing & exits
    max_pos: float = 0.33
    kelly_frac: float = 0.5
    hold_max_bars: int = 12
    tp_mult: float = 2.0
    sl_mult: float = 1.5

    # reporting
    start_cash: float = 10_000.0

cfg = CFG()

# --------------------------- Utils ---------------------------

def unify_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # case-insensitive rename
    lower = {c.lower(): c for c in d.columns}
    need = ["timestamp","symbol","open","high","low","close","volume"]
    for n in need:
        if n not in d.columns:
            for c in d.columns:
                if c.lower() == n:
                    d.rename(columns={c:n}, inplace=True)
                    break
    if "timestamp" not in d.columns: raise ValueError("Missing 'timestamp'")
    if "symbol"    not in d.columns: raise ValueError("Missing 'symbol'")
    for n in ["open","high","low","close","volume"]:
        if n not in d.columns: raise ValueError(f"Missing '{n}'")

    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open","high","low","close","volume"])
    return d

def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}T"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    out = df.set_index("timestamp").resample(rule).agg(agg).dropna().reset_index()
    return out

def sanitize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return s.fillna(0.0)

def rsi(series: pd.Series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / (down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return sanitize(out)

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return sanitize(tr.rolling(n).mean())

def vwap_band(df: pd.DataFrame, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    vwap = (tp * vol).rolling(n).sum() / (vol.rolling(n).sum() + 1e-9)
    vstd = sanitize(tp.rolling(n).std())
    upper = vwap + 1.5 * vstd
    lower = vwap - 1.5 * vstd
    return sanitize(vwap), sanitize(upper), sanitize(lower)

def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    # current return for vol proxy
    df["ret1"] = bars["close"].pct_change().fillna(0.0)
    # **forward** one-bar return used for edge & PnL
    df["fwd1"] = (bars["close"].shift(-1) / bars["close"] - 1.0)

    df["vol"]  = df["ret1"].rolling(cfg.vol_window).std().bfill().fillna(0.0)
    ema_f = bars["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
    ema_s = bars["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
    df["ema_fast"] = sanitize(ema_f)
    df["ema_slow"] = sanitize(ema_s)
    df["macd"] = sanitize(df["ema_fast"] - df["ema_slow"])
    df["rsi"] = rsi(bars["close"], cfg.rsi_window)
    df["oc"]  = ((bars["close"] - bars["open"]) / (bars["open"] + 1e-9)).fillna(0.0)
    df["hlr"] = ((bars["high"] - bars["low"]) / (bars["close"] + 1e-9)).fillna(0.0)
    df["atr"] = atr(bars, cfg.atr_win)
    vw, vu, vl = vwap_band(bars, cfg.vwap_win)
    df["vwap"], df["vwap_u"], df["vwap_l"] = vw, vu, vl
    df["zret"] = df["ret1"] / (df["vol"] + 1e-9)
    df["zmacd"] = df["macd"] / (bars["close"].rolling(cfg.ema_slow).std().bfill() + 1e-9)

    # next-bar target & drop tail NaNs
    df["target"] = (bars["close"].shift(-1) > bars["close"]).astype(int)
    df = df.dropna(subset=["fwd1","target"]).reset_index(drop=True)
    return df

FEATURE_COLS = [
    "ret1","vol","ema_fast","ema_slow","macd","rsi","oc","hlr","atr","vwap","vwap_u","vwap_l","zret","zmacd"
]

# --------------------------- Model ---------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

class CalibratedLogit:
    def __init__(self, max_iter=2000, C=0.5):
        self.cols: List[str] = []
        self.scaler: StandardScaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs", class_weight="balanced")
        self.iso = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.cols = list(X.columns)
        Xs = self.scaler.fit_transform(X.values)
        self.clf.fit(Xs, y.values)
        p = self.clf.predict_proba(Xs)[:,1]
        # guard: need some variation to fit isotonic
        if len(np.unique(np.round(p,6))) > 3:
            self.iso = IsotonicRegression(out_of_bounds="clip")
            self.iso.fit(p, y.values)
        self.fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.cols, fill_value=0.0)
        Xs = self.scaler.transform(X.values)
        p = self.clf.predict_proba(Xs)[:,1]
        if self.iso is not None:
            p = self.iso.transform(p)
        return np.clip(p, 1e-6, 1-1e-6)

# --------------------------- Simulator (uses fwd1) ---------------------------

def expected_bar_stats(y_train: pd.Series, fwd_train: pd.Series) -> Tuple[float,float]:
    pos = fwd_train[y_train==1]
    neg = fwd_train[y_train==0]
    mu_pos = float(np.nanmean(pos)) if len(pos) else 0.0  # avg up next bar
    mu_neg = float(np.nanmean(-neg)) if len(neg) else 0.0 # avg |down| next bar
    return mu_pos, mu_neg

def simulate_test(test_df: pd.DataFrame,
                  p: np.ndarray,
                  mu_pos: float,
                  mu_neg: float) -> Dict[str, float]:
    close = test_df["close"].values.astype(float)
    fwd   = test_df["fwd1"].values.astype(float)  # step return to apply
    vol   = test_df["vol"].values.astype(float)

    # expected next-bar return (fraction)
    ebar = p * mu_pos - (1.0 - p) * mu_neg
    thr = cfg.slip_roundtrip  # round-trip cost

    strength = (p - 0.5) * 2.0
    var = np.square(vol)
    var[var <= 1e-12] = np.nan
    kelly = np.clip(ebar / (var + 1e-9), -1.0, 1.0)
    kelly = np.nan_to_num(kelly, nan=0.0)
    raw_w = np.maximum(0.0, strength) * cfg.kelly_frac * np.maximum(0.0, kelly)
    target_w = np.minimum(cfg.max_pos, raw_w)
    target_w = np.where(ebar > thr, target_w, 0.0)  # honest gate

    cash = cfg.start_cash
    pos = 0.0
    entry_px = 0.0
    hold = 0
    trades = 0
    wins = 0
    trade_open = False
    eq_hist = []

    side_slip = cfg.slip_side

    for i in range(len(close)):
        w = float(target_w[i])
        v = float(vol[i])
        if not np.isfinite(v) or v <= 0:
            v = float(np.nanmedian(vol)) if np.isfinite(np.nanmedian(vol)) else 1e-4

        # rebalance cost on change
        delta = w - pos
        if abs(delta) > 1e-12:
            cash -= side_slip * abs(delta) * cash
            if w > 0 and pos == 0:
                entry_px = close[i]
                hold = 0
                trade_open = True
                trades += 1
        pos = w

        if pos > 0 and trade_open:
            hold += 1
            upnl = (close[i] - entry_px) / (entry_px + 1e-12)
            tp = cfg.tp_mult * v
            sl = -cfg.sl_mult * v
            if upnl >= tp or upnl <= sl or hold >= cfg.hold_max_bars:
                # exit -> apply side slippage on reduction
                cash *= (1.0 + pos * fwd[i])  # apply step return
                cash -= side_slip * abs(pos) * cash
                if upnl > 0: wins += 1
                pos = 0.0
                trade_open = False
                hold = 0
            else:
                cash *= (1.0 + pos * fwd[i])  # normal step
        # no else: flat, do nothing
        eq_hist.append(cash)

    # force close at end if still open
    if pos > 1e-12 and trade_open:
        cash -= side_slip * abs(pos) * cash
        if close[-1] > entry_px: wins += 1
        pos = 0.0
        trade_open = False

    start = cfg.start_cash
    pnl_pct = (cash / start - 1.0) * 100.0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

    return {
        "pnl_pct": pnl_pct,
        "trades": trades,
        "win_rate": win_rate,
        "final_cash": cash,
        "eq_hist": eq_hist,
    }

# --------------------------- Walk-forward ---------------------------

def walk_forward(dft: pd.DataFrame, sym: str, tf_min: int) -> Dict[str, float]:
    bars = resample_ohlcv(dft, tf_min)
    if len(bars) < 200:
        print(f"[{sym} {tf_min}m] Not enough data — skipping.")
        return {"pnl_pct":0,"trades":0,"win_rate":0,"splits":0,"daily":pd.Series(dtype=float)}

    feat = make_features(bars).reset_index(drop=True)

    bars_per_day = (24*60)//tf_min
    train_bars = cfg.train_days * bars_per_day
    test_bars = max(10, (cfg.test_minutes // tf_min))
    step = max(1, int(test_bars * cfg.step_fraction))

    total_pnl = 0.0
    total_trades = 0
    total_wins = 0.0
    splits = 0
    daily_pcts: List[Tuple[pd.Timestamp, float]] = []

    i = train_bars
    N = len(feat) - test_bars - 1
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | [{sym} {tf_min}m] train_bars={train_bars} test_bars={test_bars} features={len(FEATURE_COLS)}")
    while i < N:
        tr = feat.iloc[i-train_bars:i].copy()
        te = feat.iloc[i:i+test_bars].copy()

        Xtr = tr[FEATURE_COLS]
        ytr = tr["target"]
        Xte = te[FEATURE_COLS]

        model = CalibratedLogit(max_iter=2000, C=0.5)
        model.fit(Xtr, ytr)

        mu_pos, mu_neg = expected_bar_stats(ytr, tr["fwd1"])
        p = model.predict_proba(Xte)

        sim = simulate_test(te, p, mu_pos, mu_neg)
        total_pnl += sim["pnl_pct"]
        total_trades += sim["trades"]
        total_wins   += (sim["win_rate"]/100.0) * sim["trades"]
        splits += 1

        # print split line (μ in bps, thr in bps)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | INFO | [{sym} {tf_min}m] split#{splits:03d} "
              f"mu+={mu_pos*1e4:+.3f}bp mu-={mu_neg*1e4:+.3f}bp thr={cfg.slip_roundtrip*1e4:.2f}bps | "
              f"pnl={sim['pnl_pct']:+.3f}% trades={sim['trades']} win%={sim['win_rate']:.1f}% cum={total_pnl:+.3f}%")

        te_idx = bars.iloc[i:i+test_bars]["timestamp"]
        if len(te_idx):
            day_key = te_idx.iloc[-1].floor("D")
            daily_pcts.append((day_key, sim["final_cash"]/cfg.start_cash - 1.0))

        i += step

    win_rate = (total_wins/total_trades*100.0) if total_trades>0 else 0.0
    daily_df = pd.DataFrame(daily_pcts, columns=["day","ret"]).groupby("day")["ret"].mean()*100.0

    print("-"*60)
    print(f"{sym} [{tf_min}m] RESULTS (cumulative over test splits)")
    print("TF   |   CumRet%   | Trades | Win%")
    print("----------------------------------------")
    print(f"{tf_min:>2}m  | {total_pnl:+10.3f}% | {total_trades:6d} | {win_rate:5.1f}%")
    print("----------------------------------------")
    if len(daily_df):
        avg_daily_pct = float(daily_df.mean())
        avg_daily_usd = cfg.start_cash * (avg_daily_pct/100.0)
        print(f"Avg Daily P&L on $10k: {avg_daily_pct:+.3f}%  (~${avg_daily_usd:+.2f}/day)")
    print("-"*60)

    return {
        "pnl_pct": total_pnl,
        "trades": total_trades,
        "win_rate": win_rate,
        "splits": splits,
        "daily": daily_df
    }

# --------------------------- Driver ---------------------------

def load_data() -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")
    raw = pd.read_csv(cfg.csv_path)
    df = unify_cols(raw)
    df = df[df["symbol"].isin(cfg.symbols)].copy()
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    print("Starting HFT trainer (cost-aware, 0 commission, normal slippage)")
    print(f"Symbols={list(cfg.symbols)} rows={len(df):,} time={t0} → {t1}")
    print("="*60)
    return df

def run_symbol(df_all: pd.DataFrame, sym: str) -> Dict[int, Dict[str, float]]:
    print(f"SYMBOL {sym} — resampling and training")
    dft = df_all[df_all["symbol"]==sym].copy()
    out: Dict[int, Dict[str,float]] = {}
    for tf in cfg.timeframes_min:
        out[tf] = walk_forward(dft, sym, tf)
    return out

def portfolio_summary(results: Dict[str, Dict[int, Dict[str, float]]]):
    print("="*60)
    print("PORTFOLIO SUMMARY (equal-weight across symbols & timeframes)")
    rows = []
    dlist = []
    for sym, d in results.items():
        for tf, r in d.items():
            rows.append([sym, tf, r["pnl_pct"], r["trades"], r["win_rate"], r["splits"]])
            daily = r.get("daily")
            if isinstance(daily, pd.Series) and len(daily):
                dd = daily.copy(); dd.name = f"{sym}_{tf}m"; dlist.append(dd)
    if rows:
        tf_df = pd.DataFrame(rows, columns=["symbol","tf","cum_ret_pct","trades","win_rate","splits"])
        print("SYMBOL | TF | CumRet% | Trades | Win% | Splits")
        for _, row in tf_df.sort_values(["symbol","tf"]).iterrows():
            print(f"{row.symbol:>5} | {int(row.tf):>2}m | {row.cum_ret_pct:+8.3f}% | {int(row.trades):6d} | {row.win_rate:5.1f}% | {int(row.splits):3d}")
    if dlist:
        daily = pd.concat(dlist, axis=1).mean(axis=1).sort_index()
        cum = (1.0 + daily/100.0).prod() - 1.0
        avg_daily_pct = float(daily.mean())
        avg_daily_usd = cfg.start_cash * (avg_daily_pct/100.0)
        print("-"*60)
        print(f"Portfolio EW cumulative (by day): {cum*100:+.3f}%")
        print(f"Avg Daily P&L on $10k: {avg_daily_pct:+.3f}%  (~${avg_daily_usd:+.2f}/day)")
        print("-"*60)

def main():
    df = load_data()
    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for sym in cfg.symbols:
        results[sym] = run_symbol(df, sym)
    portfolio_summary(results)
    print("Trainer complete.")

if __name__ == "__main__":
    main()
