"""
FDR × Timing 验证脚本 v1.0
时机=市场在犹豫的瞬间(高熵步)，恐惧消退的种子在分叉点才塌缩
假设：FDR(恐惧消退) × 高市场熵 > FDR alone > FDR × 低市场熵
数据源：yfinance (VIX, SPY, Sector ETFs)
"""

import subprocess, sys
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yfinance', '-q'])
    import yfinance as yf

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============ CONFIG ============
SECTOR_ETFS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLRE','XLU']
VIX = '^VIX'
SPY = 'SPY'
LOOKBACK_DAYS = 1825   # 5 years for more events
FORWARD_DAYS = 10      # forward return window
VIX_SPIKE_MULT = 1.3   # VIX > 1.3x 30d mean = spike
MIN_DECAY_DAYS = 3
ENTROPY_WINDOW = 60    # rolling window for entropy percentile
ENTROPY_HIGH = 0.6     # >60th pctile = high entropy
ENTROPY_LOW = 0.4      # <40th pctile = low entropy

# ============ DOWNLOAD ============
print("="*60)
print("FDR × Timing 验证")
print("时机=市场在犹豫的瞬间，恐惧种子在分叉点塌缩")
print("="*60)
print("\nDownloading 5-year data...")

end_date = datetime.now()
start_date = end_date - timedelta(days=LOOKBACK_DAYS)

all_tickers = SECTOR_ETFS + [VIX, SPY]
raw = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True)

# Handle yfinance column format
if isinstance(raw.columns, pd.MultiIndex):
    close = raw['Close']
else:
    close = raw

vix = close[VIX].dropna()
spy = close[SPY].dropna()
sectors = close[SECTOR_ETFS].dropna()

# Align
common_idx = vix.index.intersection(spy.index).intersection(sectors.index)
vix = vix.loc[common_idx]
spy = spy.loc[common_idx]
sectors = sectors.loc[common_idx]

# Handle Series vs DataFrame
if isinstance(spy, pd.DataFrame):
    spy = spy.iloc[:, 0]
if isinstance(vix, pd.DataFrame):
    vix = vix.iloc[:, 0]

print(f"Data: {common_idx[0].date()} ~ {common_idx[-1].date()}, {len(common_idx)} days")

# ============ MARKET ENTROPY ============
print("\nComputing market entropy (cross-sectional dispersion)...")

sector_returns = sectors.pct_change().dropna()
dispersion = sector_returns.std(axis=1)

# If dispersion is DataFrame, take first column
if isinstance(dispersion, pd.DataFrame):
    dispersion = dispersion.iloc[:, 0]

# Rolling percentile rank (0~1)
def pct_rank_rolling(s, window):
    result = pd.Series(np.nan, index=s.index)
    for i in range(window-1, len(s)):
        window_data = s.iloc[i-window+1:i+1]
        result.iloc[i] = (window_data.iloc[-1] >= window_data).mean()
    return result

entropy_score = pct_rank_rolling(dispersion, ENTROPY_WINDOW)

print(f"  Entropy score: mean={entropy_score.mean():.3f}, range=[{entropy_score.min():.3f}, {entropy_score.max():.3f}]")

# ============ DETECT VIX SPIKES & DECAY ============
print("\nDetecting VIX spikes and decay rhythm...")

vix_ma30 = vix.rolling(30).mean()
vix_ratio = vix / vix_ma30

spike_mask = vix_ratio > VIX_SPIKE_MULT
spike_dates = vix.index[spike_mask]

# Group consecutive spikes into events (within 5 calendar days = same event)
events = []
current_event = []
for d in spike_dates:
    if len(current_event) == 0:
        current_event = [d]
    elif (d - current_event[-1]).days <= 5:
        current_event.append(d)
    else:
        events.append(current_event)
        current_event = [d]
if current_event:
    events.append(current_event)

print(f"  Raw spike groups: {len(events)}")

# ============ CLASSIFY DECAY & COMPUTE RESULTS ============
results = []

for evt in events:
    peak_date = evt[-1]  # last spike day
    peak_vix = vix.loc[peak_date]
    
    # VIX path after peak
    after_idx = vix.index[vix.index > peak_date][:FORWARD_DAYS]
    if len(after_idx) < MIN_DECAY_DAYS:
        continue
    
    after_vix = vix.loc[after_idx].values
    total_decay = (peak_vix - after_vix[-1]) / peak_vix
    
    if total_decay <= 0:
        continue  # VIX didn't fall
    
    # Decay rhythm: first half vs second half speed
    mid = len(after_vix) // 2
    if mid == 0:
        continue
    first_half = (peak_vix - after_vix[mid]) / peak_vix
    second_half = (after_vix[mid] - after_vix[-1]) / peak_vix
    
    # Decelerating: first half drops faster than second half
    if first_half > 0 and second_half > 0:
        ratio = first_half / second_half
        if ratio > 1.2:
            rhythm = 'decelerating'
        elif ratio < 0.8:
            rhythm = 'accelerating'
        else:
            rhythm = 'linear'
    elif first_half > 0 and second_half <= 0:
        rhythm = 'decelerating'  # first half drops, second half bounces = decel
    else:
        rhythm = 'other'
    
    # Forward SPY return
    spy_after_idx = spy.index[spy.index >= peak_date][:FORWARD_DAYS+1]
    if len(spy_after_idx) < FORWARD_DAYS + 1:
        continue
    spy_after = spy.loc[spy_after_idx]
    fwd_return = (spy_after.iloc[-1] - spy_after.iloc[0]) / spy_after.iloc[0]
    
    # Entropy at trigger
    if peak_date in entropy_score.index:
        ent = entropy_score.loc[peak_date]
    else:
        nearby = entropy_score[entropy_score.index <= peak_date]
        if len(nearby) == 0:
            continue
        ent = nearby.iloc[-1]
    
    if pd.isna(ent):
        continue
    
    results.append({
        'date': peak_date,
        'peak_vix': peak_vix,
        'vix_ratio': vix_ratio.loc[peak_date] if peak_date in vix_ratio.index else np.nan,
        'total_decay': total_decay,
        'rhythm': rhythm,
        'entropy': ent,
        'fwd_return': fwd_return,
        'spy_fwd_pct': fwd_return * 100,
    })

df = pd.DataFrame(results)
print(f"  Valid FDR events: {len(df)}")

if len(df) == 0:
    print("\nNo events found. Try: increase LOOKBACK_DAYS or lower VIX_SPIKE_MULT.")
    sys.exit(1)

# ============ ANALYSIS ============
print("\n" + "="*60)
print("1. BASELINE: FDR rhythms (original factor, no timing)")
print("="*60)

for rhythm in ['decelerating', 'accelerating', 'linear', 'other']:
    sub = df[df['rhythm'] == rhythm]
    if len(sub) == 0:
        continue
    wr = (sub['fwd_return'] > 0).mean()
    mr = sub['fwd_return'].mean() * 100
    print(f"  {rhythm:14s}: n={len(sub):2d}, win={wr:.1%}, mean_ret={mr:+.2f}%")

print("\n" + "="*60)
print("2. TIMING TEST: Decelerating × Market Entropy")
print("="*60)
print("   High entropy = market在犹豫(分叉点) = 时机到了")
print("   Low entropy  = market有共识(趋势中) = 时机没到")

decel = df[df['rhythm'] == 'decelerating']
decel_high = decel[decel['entropy'] > ENTROPY_HIGH]
decel_mid = decel[(decel['entropy'] >= ENTROPY_LOW) & (decel['entropy'] <= ENTROPY_HIGH)]
decel_low = decel[decel['entropy'] < ENTROPY_LOW]

print(f"\n  {'Group':<30s} {'n':>3s} {'WinRate':>8s} {'MeanRet':>10s}")
print(f"  {'-'*55}")

for label, sub in [("HIGH entropy (timing right)", decel_high),
                    ("MID entropy", decel_mid),
                    ("LOW entropy (timing wrong)", decel_low),
                    ("ALL decelerating (baseline)", decel)]:
    if len(sub) > 0:
        wr = (sub['fwd_return'] > 0).mean()
        mr = sub['fwd_return'].mean() * 100
        print(f"  {label:<30s} {len(sub):3d} {wr:>7.1%} {mr:>+9.2f}%")
    else:
        print(f"  {label:<30s}   0     N/A       N/A")

print("\n" + "="*60)
print("3. FULL MATRIX: Rhythm × Entropy")
print("="*60)

for rhythm in ['decelerating', 'accelerating', 'linear']:
    sub = df[df['rhythm'] == rhythm]
    if len(sub) == 0:
        continue
    print(f"\n  {rhythm.upper()}:")
    for ent_label, ent_mask in [("HIGH", sub['entropy'] > ENTROPY_HIGH),
                                  ("MID", (sub['entropy'] >= ENTROPY_LOW) & (sub['entropy'] <= ENTROPY_HIGH)),
                                  ("LOW", sub['entropy'] < ENTROPY_LOW)]:
        s = sub[ent_mask]
        if len(s) > 0:
            wr = (s['fwd_return'] > 0).mean()
            mr = s['fwd_return'].mean() * 100
            print(f"    entropy={ent_label}: n={len(s):2d}, win={wr:.1%}, ret={mr:+.2f}%")

# ============ EVENT DETAIL ============
print("\n" + "="*60)
print("4. EVENT DETAIL (decelerating only)")
print("="*60)
print(f"  {'Date':<12s} {'VIX':>5s} {'Ratio':>6s} {'Decay':>6s} {'Entropy':>8s} {'Label':>5s} {'FwdRet':>8s}")
print(f"  {'-'*55}")

for _, row in decel.sort_values('date').iterrows():
    ent_label = "HIGH" if row['entropy'] > ENTROPY_HIGH else ("LOW" if row['entropy'] < ENTROPY_LOW else "MID")
    print(f"  {row['date'].strftime('%Y-%m-%d'):<12s} {row['peak_vix']:5.1f} "
          f"{row['vix_ratio']:5.2f}x {row['total_decay']:5.1%} "
          f"{row['entropy']:7.3f} {ent_label:>5s} {row['spy_fwd_pct']:+7.2f}%")

# ============ VERDICT ============
print("\n" + "="*60)
print("★ TIMING HYPOTHESIS VERDICT")
print("="*60)
print("  D10映射: 10%种子在高熵步塌缩→60%现实")
print("  市场映射: FDR种子在高熵市场塌缩→更高胜率")
print()

if len(decel_high) >= 3 and len(decel_low) >= 3:
    wr_high = (decel_high['fwd_return'] > 0).mean()
    wr_low = (decel_low['fwd_return'] > 0).mean()
    gap = wr_high - wr_low
    ret_gap = decel_high['fwd_return'].mean() - decel_low['fwd_return'].mean()
    
    if gap > 0.15:
        print(f"  ★★★ TIMING CONFIRMED: win rate gap = +{gap:.1%}")
        print(f"  ★★★ Return gap = +{ret_gap*100:.2f}%")
        print(f"  → FDR种子在高熵步塌缩，时机对了效果翻倍")
    elif gap > 0.05:
        print(f"  ★★ TIMING SUPPORTED: gap = +{gap:.1%}, 样本偏小需更多验证")
    elif gap > 0:
        print(f"  ★ TIMING WEAK: gap = +{gap:.1%}")
    else:
        print(f"  ✗ TIMING NOT CONFIRMED: gap = {gap:.1%}")
        print(f"  → 可能5年数据不够，或截面离散度不是最佳熵度量")
elif len(decel) >= 3:
    print(f"  ⚠ 样本不足: high={len(decel_high)}, low={len(decel_low)}")
    print(f"  → 尝试: 降低ENTROPY_HIGH/LOW阈值, 或增大LOOKBACK_DAYS")
else:
    print(f"  ⚠ 减速消退事件太少({len(decel)}), 无法验证")

print("\nDone.")
