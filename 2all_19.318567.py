# -*- coding: utf-8 -*-
"""
Unified OIBC irradiance pipeline (patched: items 1,2,3,5,6)

Changes applied:
 1) Per-PV clear-sky calibration (alpha_pv) and kt recompute
 2) Use kt (not nins) for lag/rolling features
 3) Increase ELEV_MIN to 9 degrees (stronger twilight cut)
 5) Expand lag/rolling specs with multi-scale windows and std
 6) Blend model kt_hat with persistence (kt_lag1) at inference

- Input: BASE/train.csv
- Saves:
  * test_pv_ids.csv
  * pred_results.csv  (time, pv_id, nins, pred_nins, abs_error)
  * pipeline_run.log

Note: "train_processed_full.csv" write is kept commented (can be enabled as desired).
"""

import pandas as pd
import numpy as np
import time as t
import traceback, gc, random, logging, sys
from pathlib import Path
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# =========================
# 0) Paths / Parameters
# =========================
BASE = Path("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA")  # adjust to your env
CSV_PATH = BASE / "train.csv"

SAVE_PROCESSED_FULL = BASE / "train_processed_full.csv"   # optional (disabled below)
SAVE_TEST_PV_IDS    = BASE / "test_pv_ids.csv"
SAVE_PRED_RESULT    = BASE / "pred_results.csv"
SAVE_LOG            = BASE / "pipeline_run.log"

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Test split config
TEST_BY_RATIO   = True
TEST_RATIO      = 0.12
TEST_NUM_PV     = 27

# Exclude PV numbers (by numeric suffix in pv_id)
EXCLUDE_PV_NUMBERS = [7, 8, 10, 23, 29, 39, 48, 49, 64, 72, 78, 82, 95, 114, 117, 121,
                      122, 134, 165, 173, 175, 180, 182, 183, 192, 194, 204]

# Lag/Rolling (5-min step assumed)
# (5) Multi-scale expansion incl. std
LAG_STEPS   = [1, 2, 3, 6, 12, 24]  # up to 2 hours
ROLL_SPECS  = [(3, "mean"), (6, "mean"), (12, "mean"), (36, "mean"), (12, "std"), (36, "std")]
ROLL_MIN_PERIODS = 1

# (3) Stronger twilight cut during training
ELEV_MIN = 9.0

# (6) Persistence blend weight
PERSIST_MIX_W = 0.20  # 0.1~0.3 recommended

# =========================
# Logging (console + file)
# =========================
logger = logging.getLogger("oibc_pipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(SAVE_LOG, encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

def log(msg: str):
    print(msg)
    logger.info(msg)

# =========================
# Utils: memory downcast
# =========================

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype(np.int32)
    return df

# =========================
# 1) Load & Datetime
# =========================

def load_data(path: Path) -> pd.DataFrame:
    t0 = t.time()
    df = pd.read_csv(path)
    log(f"[IO] load: {t.time()-t0:.2f}s, shape={df.shape}")
    if "time" not in df.columns or "pv_id" not in df.columns:
        raise ValueError("필수 컬럼(time, pv_id)이 없습니다.")
    t1 = t.time()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    log(f"[Time] parsed in {t.time()-t1:.2f}s, time_na={df['time'].isna().sum()}")
    return df

# =========================
# 2) Imputation rules
# =========================

cubic_cols = [
    "temp_a","temp_b","appr_temp","real_feel_temp","real_feel_temp_shade",
    "wind_chill_temp","temp_max","temp_min","dew_point","rel_hum",
    "humidity","pressure","ground_press","vis","uv_idx"
]
step_ffill_cols = ["precip_1h","rain","snow"]
exclude_from_interp = {"time","pv_id","type","energy","nins"}

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def impute_by_rules(group_df: pd.DataFrame, pv_id: str) -> pd.DataFrame:
    log(f"\n▶ [Group] {pv_id} rows={len(group_df)}")
    g = group_df.sort_values("time")

    present_cubic = [c for c in cubic_cols if c in g.columns]
    present_step  = [c for c in step_ffill_cols if c in g.columns]
    g = _coerce_numeric(g, present_cubic + present_step)

    # cubic spline (fallback: polynomial)
    if present_cubic:
        try:
            log(f"  - cubic_spline: {present_cubic[:6]}{' ...' if len(present_cubic)>6 else ''}")
            g.loc[:, present_cubic] = g.loc[:, present_cubic].interpolate(
                method="spline", order=3, limit_direction="both"
            )
        except Exception as e:
            log(f"  [WARN] spline fail → polynomial fallback: {e}")
            traceback.print_exc(limit=1)
            g.loc[:, present_cubic] = g.loc[:, present_cubic].interpolate(
                method="polynomial", order=3, limit_direction="both"
            )

    if present_step:
        log(f"  - step_ffill: {present_step}")
        # keep original unlimited ffill (item 4 is NOT applied here per user's selection)
        g.loc[:, present_step] = g.loc[:, present_step].ffill()

    # linear for the rest numeric
    numeric_cols = g.select_dtypes(include=[np.number]).columns.tolist()
    other_linear_cols = [
        c for c in numeric_cols
        if c not in present_cubic and c not in present_step and c not in exclude_from_interp
    ]
    if other_linear_cols:
        log(f"  - linear ({len(other_linear_cols)}): {other_linear_cols[:6]}{' ...' if len(other_linear_cols)>6 else ''}")
        try:
            g.loc[:, other_linear_cols] = g.loc[:, other_linear_cols].interpolate(
                method="linear", limit_direction="both"
            )
        except Exception as e:
            log(f"  [ERROR] linear interp: {e}")
            traceback.print_exc(limit=1)

    log(f"  ✓ done: {pv_id}")
    return g

def impute_all(df: pd.DataFrame) -> pd.DataFrame:
    if "pv_id" not in df.columns:
        raise ValueError("pv_id 컬럼이 필요합니다.")
    pvs = df["pv_id"].unique().tolist()
    log(f"[Impute] groups={len(pvs)}")
    imputed_list = []
    for i, pv in enumerate(pvs, 1):
        try:
            imputed_list.append(impute_by_rules(df[df["pv_id"] == pv], pv))
        except Exception as e:
            log(f"[ERROR] impute group {pv}: {e}")
            traceback.print_exc(limit=1)
        if i % 5 == 0 or i == len(pvs):
            log(f"  -> progress {i}/{len(pvs)} ({i/len(pvs)*100:.1f}%)")
    out = pd.concat(imputed_list, ignore_index=True)
    out = downcast_numeric(out)
    log("[Mem] after impute+downcast: " +
        str(round(out.memory_usage(deep=True).sum()/1024**2, 1)) + " MB")
    return out

# =========================
# 3) Time & Solar features
# =========================

KOR_LAT = 36.5
KOR_LON = 127.9

def add_time_and_solar_features(df: pd.DataFrame, tz="Asia/Seoul") -> pd.DataFrame:
    g = df.copy()
    if pd.api.types.is_datetime64_any_dtype(g["time"]):
        if g["time"].dt.tz is None:
            g["time"] = g["time"].dt.tz_localize(tz)
        else:
            g["time"] = g["time"].dt.tz_convert(tz)
    else:
        g["time"] = pd.to_datetime(g["time"], errors="coerce").dt.tz_localize(tz)

    g["hour"] = g["time"].dt.hour
    g["minute"] = g["time"].dt.minute
    g["doy"] = g["time"].dt.dayofyear

    day_angle  = 2*np.pi*((g["hour"]*60 + g["minute"]) / (24*60))
    g["sin_time"] = np.sin(day_angle); g["cos_time"] = np.cos(day_angle)
    year_angle = 2*np.pi*(g["doy"] / 365.25)
    g["sin_doy"] = np.sin(year_angle); g["cos_doy"] = np.cos(year_angle)

    lat = np.deg2rad(KOR_LAT); lon_deg = KOR_LON
    local_minutes = (g["hour"]*60 + g["minute"]).astype(float)
    gamma = 2*np.pi*((g["doy"] - 1 + (local_minutes/1440.0)) / 365.0)
    eqt = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                  - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    lon_correction_min = 4.0 * (lon_deg - 135.0)
    time_offset = eqt + lon_correction_min
    tst = local_minutes + time_offset
    hra = np.deg2rad((tst/4.0) - 180.0)
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))
    cos_zenith = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(hra)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.arccos(cos_zenith)
    solar_elev_deg = 90.0 - np.rad2deg(zenith)
    g["solar_elev_deg"] = solar_elev_deg.astype(np.float32)
    g["is_day"] = (solar_elev_deg > 0.0).astype(np.int8)

    g["eqt"]  = eqt.astype(np.float32)
    g["hra"]  = ((tst/4.0) - 180.0).astype(np.float32)
    g["decl"] = np.rad2deg(decl).astype(np.float32)
    return g

# =========================
# 4) Row-wise features & dynamics & lag/rolling (generic)
# =========================

def add_rowwise_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    for c in ["solar_elev_deg","temp_a","dew_point","vis","real_feel_temp",
              "real_feel_temp_shade","appr_temp","wind_chill_temp",
              "ground_press","pressure","uv_idx","rain","snow","precip_1h"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    if "solar_elev_deg" in g.columns:
        elev_rad = np.deg2rad(g["solar_elev_deg"].astype(float))
        cosZ = np.sin(elev_rad)
        g["cosZ_pos"] = np.clip(cosZ, 0.0, None).astype(np.float32)
        g["sin_elev"] = np.sin(np.clip(elev_rad, 0, None)).astype(np.float32)
        g["elev_sq"]  = (g["solar_elev_deg"].clip(lower=0)**2).astype(np.float32)

    for c in ["decl","hra","eqt"]:
        if c in g.columns:
            g[c] = g[c].astype(np.float32)

    if {"temp_a","dew_point"} <= set(g.columns):
        g["dewpoint_depr"] = (g["temp_a"] - g["dew_point"]).astype(np.float32)

    if "vis" in g.columns:
        g["vis_inv"] = (1.0 / (g["vis"].abs() + 1e-3)).astype(np.float32)
        g["vis_log"] = np.log1p(g["vis"].clip(lower=0)).astype(np.float32)

    if {"ground_press","pressure"} <= set(g.columns):
        g["press_diff"] = (g["ground_press"] - g["pressure"]).astype(np.float32)

    if {"real_feel_temp","temp_a"} <= set(g.columns):
        g["realfeel_gap"] = (g["real_feel_temp"] - g["temp_a"]).astype(np.float32)
    if {"real_feel_temp_shade","temp_a"} <= set(g.columns):
        g["shade_gap"] = (g["real_feel_temp_shade"] - g["temp_a"]).astype(np.float32)
    if {"appr_temp","temp_a"} <= set(g.columns):
        g["appr_gap"] = (g["appr_temp"] - g["temp_a"]).astype(np.float32)
    if {"wind_chill_temp","temp_a"} <= set(g.columns):
        g["windchill_gap"] = (g["wind_chill_temp"] - g["temp_a"]).astype(np.float32)

    rain = g["rain"] if "rain" in g.columns else 0
    snow = g["snow"] if "snow" in g.columns else 0
    precip = g["precip_1h"] if "precip_1h" in g.columns else 0
    g["precip_flag"] = ((rain>0) | (snow>0) | (precip>0)).astype(np.int8)

    wet_code = np.zeros(len(g), dtype=np.int8)
    if "rain" in g.columns: wet_code = np.where(g["rain"]>0, 1, wet_code)
    if "snow" in g.columns: wet_code = np.where(g["snow"]>0, 2, wet_code)
    g["wet_code"] = wet_code

    if {"uv_idx","cosZ_pos"} <= set(g.columns):
        g["uv_norm"] = (g["uv_idx"] / (g["cosZ_pos"] + 1e-3)).astype(np.float32)
    return g

def add_weather_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values(["pv_id","time"]).copy()
    def _lag(col, L):
        return g.groupby("pv_id", sort=False)[col].shift(L)

    for col in ["vis","uv_idx","dewpoint_depr","pressure","ground_press"]:
        if col in g.columns:
            g[f"{col}_diff1"] = g[col] - _lag(col, 1)
            g[f"{col}_roll6_mean"]  = g.groupby("pv_id", sort=False)[col].shift(1)\
                                        .rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
            g[f"{col}_roll12_mean"] = g.groupby("pv_id", sort=False)[col].shift(1)\
                                        .rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)

    if "precip_flag" in g.columns:
        pf = g["precip_flag"]
        g["precip_onset"]  = ((pf==1) & (_lag("precip_flag",1)==0)).astype(np.int8)
        g["precip_offset"] = ((pf==0) & (_lag("precip_flag",1)==1)).astype(np.int8)
    return g

def add_lag_rolling(df: pd.DataFrame,
                    by="pv_id", time_col="time", target="nins",
                    lag_steps=(1,2,3,6,12),
                    roll_specs=((3,"mean"), (12,"mean")),
                    roll_min_periods=1) -> pd.DataFrame:
    g = df.sort_values([by, time_col]).copy()
    for L in lag_steps:
        g[f"{target}_lag{L}"] = g.groupby(by, sort=False)[target].shift(L)
    for win, agg in roll_specs:
        base = g.groupby(by, sort=False)[target].shift(1)
        rolled = base.groupby(g[by], sort=False)\
                     .rolling(window=win, min_periods=roll_min_periods).agg(agg)
        g[f"{target}_roll{win}_{agg}"] = rolled.reset_index(level=0, drop=True)
    return g

# =========================
# 5) Clear-sky & kt (with per-PV calibration) — (1)
# =========================

def compute_clear_sky_haurwitz(df, elev_col="solar_elev_deg"):
    cs = np.zeros(len(df), dtype=np.float32)
    if elev_col in df.columns:
        elev_rad = np.deg2rad(df[elev_col].astype(float))
        cosZ = np.clip(np.sin(elev_rad), 0.0, None)
        mask = cosZ > 0
        tmp = np.zeros_like(cosZ, dtype=np.float64)
        tmp[mask] = 1098.0 * cosZ[mask] * np.exp(-0.057 / np.clip(cosZ[mask], 1e-6, None))
        cs = tmp.astype(np.float32)
    return cs


def calibrate_clear_sky_per_pv(df: pd.DataFrame, elev_col="solar_elev_deg", q=0.95) -> pd.DataFrame:
    """
    Estimate per-PV alpha to scale clear-sky I_cs and recompute kt.
    Steps:
      * compute raw I_cs (Haurwitz)
      * temporary kt = nins / I_cs_raw
      * pick "clearest" subset per PV (top q quantile within day & no-precip & high visibility)
      * alpha_pv = median(nins / I_cs_raw) on that subset, clipped [0.7, 1.3]
      * I_cs = alpha_pv * I_cs_raw; kt = nins / I_cs
    """
    g = df.copy()
    g["I_cs_raw"] = compute_clear_sky_hauritz(g, elev_col) if False else compute_clear_sky_haurwitz(g, elev_col)
    eps = 1e-3
    tmp_kt = (g["nins"] / (g["I_cs_raw"] + eps)).clip(0, 2.0)

    alphas = {}
    for pv, sub in g.groupby("pv_id"):
        mask = (sub["is_day"] == 1)
        if "precip_flag" in sub.columns:
            mask &= (sub["precip_flag"] == 0)
        if "vis" in sub.columns and sub["vis"].notna().any():
            try:
                vthr = sub["vis"].quantile(0.8)
                mask &= (sub["vis"] >= vthr)
            except Exception:
                pass

        if mask.sum() < 50:
            cand = sub
        else:
            hi = tmp_kt[sub.index][mask]
            thresh = hi.quantile(q)
            cand = sub.loc[hi.index[hi >= thresh]]

        num = cand["nins"].astype(float)
        den = (cand["I_cs_raw"] + eps).astype(float)
        alpha = float(np.median(num / den)) if len(cand) else 1.0
        alphas[pv] = np.clip(alpha, 0.7, 1.3)

    g["alpha_pv"] = g["pv_id"].map(alphas).astype(np.float32)
    g["I_cs"] = (g["I_cs_raw"] * g["alpha_pv"]).astype(np.float32)
    g["kt"] = (g["nins"] / (g["I_cs"] + eps)).clip(0.0, 1.4).astype(np.float32)
    return g

# =========================
# 6) Split helpers
# =========================

def pick_test_pv_ids(all_pv_ids):
    nums = []
    for s in all_pv_ids:
        try:
            nums.append(int(str(s).split("_")[-1]))
        except:
            pass
    pairs = list(zip(all_pv_ids, nums))
    avail = [(pid, n) for pid, n in pairs if n not in EXCLUDE_PV_NUMBERS]
    if not avail:
        base_list = list(all_pv_ids)
    else:
        base_list = [pid for pid, _ in avail]

    if TEST_BY_RATIO:
        k = max(1, int(len(base_list) * TEST_RATIO))
    else:
        k = min(max(1, TEST_NUM_PV), len(base_list))
    sampled = sorted(random.sample(base_list, k))
    return sampled

# =========================
# 7) Main
# =========================

def main():
    start_all = t.time()
    log("===== Unified OIBC pipeline start (patched 1,2,3,5,6) =====")

    # A) Load & Impute
    data = load_data(CSV_PATH)
    data_imputed = impute_all(data)

    # B) Solar features
    t0 = t.time()
    data_feat = add_time_and_solar_features(data_imputed, tz="Asia/Seoul")
    log("[Solar] features added: hour/minute/doy, sin/cos(time,doy), solar_elev_deg, is_day, eqt/hra/decl")
    log(f"[Solar] done in {t.time()-t0:.2f}s")

    # C) Row-wise & Dynamics
    data_feat = add_rowwise_engineered_features(data_feat)
    data_feat = add_weather_dynamics(data_feat)

    # D) Clear-sky calibration & kt (1)
    data_feat = calibrate_clear_sky_per_pv(data_feat, elev_col="solar_elev_deg", q=0.95)

    # E) kt-based lag/rolling (2 + 5)
    data_feat = add_lag_rolling(
        data_feat,
        by="pv_id", time_col="time", target="kt",
        lag_steps=LAG_STEPS, roll_specs=ROLL_SPECS, roll_min_periods=ROLL_MIN_PERIODS
    )

    # Optional: persist full processed (disabled by default)
    # data_feat.to_csv(SAVE_PROCESSED_FULL, index=False, encoding="utf-8-sig")
    # log(f"✅ 전체 파생 포함 데이터 저장: {SAVE_PROCESSED_FULL}")

    # F) Train/Valid/Test split
    all_pv_ids = sorted(data_feat["pv_id"].unique().tolist())
    test_pv_ids = pick_test_pv_ids(all_pv_ids)
    pd.DataFrame({"pv_id": test_pv_ids}).to_csv(SAVE_TEST_PV_IDS, index=False, encoding="utf-8-sig")
    log(f"[Split] test PV count={len(test_pv_ids)} → {SAVE_TEST_PV_IDS}")

    mask_test = data_feat["pv_id"].isin(test_pv_ids)
    # Train on day & sufficiently high sun elevation (3)
    train_mask_base = (~mask_test) & (data_feat["is_day"] == 1) & (data_feat["solar_elev_deg"] >= ELEV_MIN)

    # Validation: hold out some train PVs
    train_pvs = sorted(data_feat.loc[train_mask_base, "pv_id"].unique().tolist())
    n_val_pv  = max(1, int(len(train_pvs) * 0.12))
    val_pvs   = set(random.sample(train_pvs, n_val_pv))
    val_mask  = train_mask_base & data_feat["pv_id"].isin(val_pvs)
    train_mask= train_mask_base & ~data_feat["pv_id"].isin(val_pvs)

    log(f"[Rows] train={train_mask.sum()}, valid={val_mask.sum()}, test={mask_test.sum()}")

    # G) Feature selection
    exclude_cols = ["time","pv_id","type","energy","nins","is_day","minute","kt"]  # do not leak current kt
    feature_candidates = [c for c in data_feat.columns if c not in exclude_cols]
    features = [c for c in feature_candidates if np.issubdtype(data_feat[c].dtype, np.number)]
    log(f"[Feat] {len(features)} features (e.g., {features[:12]}{' ...' if len(features)>12 else ''})")

    # H) Numpy views (train on kt)
    X_tr = data_feat.loc[train_mask, features].to_numpy(dtype=np.float32, copy=False)
    y_tr = data_feat.loc[train_mask, "kt"].to_numpy(copy=False)
    X_va = data_feat.loc[val_mask,   features].to_numpy(dtype=np.float32, copy=False)
    y_va = data_feat.loc[val_mask,   "kt"].to_numpy(copy=False)
    X_te = data_feat.loc[mask_test,  features].to_numpy(dtype=np.float32, copy=False)
    y_te = data_feat.loc[mask_test,  "nins"].to_numpy(copy=False)  # evaluate on nins

    # Clean NaNs from new kt lags/rollings
    keep_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    if not keep_tr.all():
        log(f"[Clean] drop {(~keep_tr).sum()} rows in train (NaN by lags/rollings)")
        X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr]
    keep_va = np.isfinite(X_va).all(axis=1) & np.isfinite(y_va)
    if not keep_va.all():
        log(f"[Clean] drop {(~keep_va).sum()} rows in valid (NaN by lags/rollings)")
        X_va, y_va = X_va[keep_va], y_va[keep_va]

    # I) Train LightGBM on kt
    log("[Train] LightGBM (kt target) start")
    lgbm = LGBMRegressor(
        objective="regression_l1",
        random_state=SEED,
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=200,
        subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=400),
            lgb.log_evaluation(period=200)
        ]
    )

    # J) Predict: kt_hat → blend with persistence (6) → to nins; apply night=0, nonneg
    log("[Predict] on test")
    best_iter = getattr(lgbm, "best_iteration_", None)
    kt_hat = lgbm.predict(X_te, num_iteration=best_iter).astype(np.float32)

    # persistence
    kt_persist = data_feat.loc[mask_test, "kt_lag1"].to_numpy(dtype=np.float32)
    kt_blend = (1.0 - PERSIST_MIX_W) * kt_hat + PERSIST_MIX_W * np.nan_to_num(kt_persist, nan=kt_hat)

    I_cs_te = data_feat.loc[mask_test, "I_cs"].to_numpy(copy=False).astype(np.float32)
    pred = (kt_blend * I_cs_te).astype(np.float32)

    is_day_test = data_feat.loc[mask_test, "is_day"].to_numpy(copy=False)
    pred[is_day_test == 0] = 0.0
    pred = np.where(pred < 0, 0, pred)

    # K) Evaluate & save
    mae = mean_absolute_error(y_te, pred)
    log(f"[Eval] MAE (kt-model + per-PV I_cs + blend + night=0): {mae:.6f}")

    result_cols = ["pv_id","time","nins"]
    result_df = data_feat.loc[mask_test, result_cols].copy()
    result_df["pred_nins"] = pred.astype(np.float32)
    result_df["abs_error"] = (result_df["nins"] - result_df["pred_nins"]).abs()
    result_df.to_csv(SAVE_PRED_RESULT, index=False, encoding="utf-8-sig")
    log(f"✅ 예측 결과 저장: {SAVE_PRED_RESULT}")
    log("컬럼 예시: ['pv_id','time','nins','pred_nins','abs_error']")

    # L) Feature importance (top-30)
    try:
        importances = lgbm.feature_importances_
        fi = pd.DataFrame({"feature": features, "importance": importances}) \
               .sort_values("importance", ascending=False).reset_index(drop=True)
        log("\n[Top-30 Feature Importances]")
        for i, row in fi.head(30).iterrows():
            log(f"{i+1:2d}. {row['feature']}: {int(row['importance'])}")
    except Exception as e:
        log(f"[WARN] feature importance 표시 실패: {e}")

    # M) Cleanup
    del data, data_imputed, data_feat, X_tr, X_va, X_te, y_tr, y_va, y_te, kt_hat, pred
    gc.collect()

    log(f"\n[All Done] total {t.time()-start_all:.2f}s")
    log("===== Unified OIBC pipeline end =====")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        traceback.print_exc(limit=2)
