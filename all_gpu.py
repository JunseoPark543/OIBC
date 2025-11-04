# -*- coding: utf-8 -*-
"""
OIBC irradiance 2-stage pipeline (kt target + pseudo kt-lag from predictions, GPU 지원 버전)

입력:
  - BASE/train.csv  (nins 있음)
  - BASE/test.csv   (nins 없음, 또는 전부 결측)

출력:
  - BASE/submission_two_stage.csv  (time, pv_id, nins 형식)
  - BASE/pipeline_two_stage.log    (로그)

구조:
  1) train/test 로드 + concat
  2) pv_id별 보간 (cubic/step/linear) + 메모리 downcast
  3) 시간/태양/rowwise/dynamics 피처 생성
  4) Haurwitz clear-sky I_cs + per-PV alpha 보정 + train만 kt 생성
  5) [1단계] exogenous 피처만으로 kt 예측 (GroupKFold OOF)
         → train: kt_hat_stage1(OOF), test: kt_hat_stage1(mean of folds)
  6) kt_hat_stage1 기준 lag/rolling 생성 (nins 없이도 가능)
  7) [2단계]  kt_hat_stage1 + lag/rolling + 기타 피처로 kt 재학습
         → 예측한 kt_hat_final * I_cs → nins
  8) night=0, 음수컷 후 제출 파일 생성
"""

import pandas as pd
import numpy as np
import time as t
import traceback, gc, random, logging, sys
from pathlib import Path

from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

# =========================
# LightGBM device config (GPU / CPU)
# =========================

# GPU를 쓸 수 있으면 True, GPU 빌드가 아니거나 에러 나면 False로 바꿔서 CPU 모드로 사용
USE_GPU = True

def get_lgbm_device_params():
    if USE_GPU:
        # GPU 모드 (RTX 시리즈 등 있을 때)
        return dict(
            device_type="gpu",
            gpu_platform_id=0,
            gpu_device_id=0
        )
    else:
        # 안전한 CPU 모드 (윈도우 세그폴트 방지 세팅 포함)
        return dict(
            device_type="cpu",
            n_jobs=1,
            force_row_wise=True
        )

LGBM_DEVICE_PARAMS = get_lgbm_device_params()

# =========================
# 0) Paths / Parameters
# =========================

# ★★ 피시방 / 로컬 환경에 맞게 수정 ★★
BASE = Path("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA")

TRAIN_CSV = BASE / "train.csv"
TEST_CSV  = BASE / "test.csv"

SUBMISSION_PATH = BASE / "submission_two_stage.csv"
LOG_PATH        = BASE / "pipeline_two_stage.log"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Stage1 kt_hat lag/rolling 설정 (5분 단위라 가정)
LAG_STEPS   = [1, 2, 3, 6, 12, 24]  # 최대 2시간
ROLL_SPECS  = [(3, "mean"), (6, "mean"), (12, "mean"),
               (36, "mean"), (12, "std"), (36, "std")]
ROLL_MIN_PERIODS = 1

# 학습 시 태양고도 컷
ELEV_MIN = 9.0

# Stage1에서 사용할 최대 학습 행 수 (메모리 보호용)
MAX_STAGE1_ROWS = 800_000

# =========================
# Logging
# =========================

logger = logging.getLogger("oibc_two_stage")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

def log(msg: str):
    print(msg)
    logger.info(msg)

# =========================
# Utils
# =========================

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype(np.int32)
    return df

def load_data(path: Path) -> pd.DataFrame:
    t0 = t.time()
    df = pd.read_csv(path)
    log(f"[IO] load {path.name}: {t.time()-t0:.2f}s, shape={df.shape}")
    if "time" not in df.columns or "pv_id" not in df.columns:
        raise ValueError("필수 컬럼(time, pv_id)이 없습니다.")
    t1 = t.time()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    log(f"[Time] parsed in {t.time()-t1:.2f}s, time_na={df['time'].isna().sum()}")
    return df

# =========================
# 1) Imputation rules
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
        g.loc[:, present_step] = g.loc[:, present_step].ffill()

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
# 2) Time & Solar features
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
    g["sin_time"] = np.sin(day_angle)
    g["cos_time"] = np.cos(day_angle)
    year_angle = 2*np.pi*(g["doy"] / 365.25)
    g["sin_doy"] = np.sin(year_angle)
    g["cos_doy"] = np.cos(year_angle)

    lat = np.deg2rad(KOR_LAT)
    lon_deg = KOR_LON
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
# 3) Row-wise features & dynamics
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
    if "rain" in g.columns:
        wet_code = np.where(g["rain"]>0, 1, wet_code)
    if "snow" in g.columns:
        wet_code = np.where(g["snow"]>0, 2, wet_code)
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
                    by="pv_id", time_col="time", target="kt_hat_stage1",
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
# 4) Clear-sky (Haurwitz)
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

# =========================
# 5) Main
# =========================

def main():
    start_all = t.time()
    log("===== Two-stage kt pipeline (GPU-ready) start =====")

    # A) Load train/test
    train = load_data(TRAIN_CSV)
    test  = load_data(TEST_CSV)

    # test에 nins가 없으면 NaN 컬럼 생성
    if "nins" not in test.columns:
        test["nins"] = np.nan

    train["is_train"] = 1
    test["is_train"]  = 0

    # 컬럼 align (없는 컬럼은 자동으로 NaN으로 채워짐)
    all_cols = sorted(set(train.columns) | set(test.columns))
    train = train.reindex(columns=all_cols)
    test  = test.reindex(columns=all_cols)

    data = pd.concat([train, test], ignore_index=True)
    del train, test
    gc.collect()

    # B) Imputation (pv_id별)
    data_imputed = impute_all(data)
    del data
    gc.collect()

    # C) Time & Solar
    t0 = t.time()
    data_feat = add_time_and_solar_features(data_imputed, tz="Asia/Seoul")
    log("[Solar] 시간/태양 피처 추가 완료")
    log(f"[Solar] done in {t.time()-t0:.2f}s")

    # D) Row-wise & Dynamics
    data_feat = add_rowwise_engineered_features(data_feat)
    data_feat = add_weather_dynamics(data_feat)

    # E) Clear-sky I_cs_raw + per-PV alpha (train만 사용해 보정)
    data_feat["I_cs_raw"] = compute_clear_sky_haurwitz(data_feat, "solar_elev_deg")

    # train mask
    is_train = data_feat["is_train"] == 1

    # nins numeric(float)
    data_feat["nins"] = pd.to_numeric(data_feat["nins"], errors="coerce")

    # per-PV alpha 추정 (train + 주간 위주)
    eps = 1e-3
    alphas = {}
    train_sub = data_feat[is_train].copy()

    for pv, sub in train_sub.groupby("pv_id"):
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
            # raw kt로 clear subset 잡기
            num = sub.loc[mask, "nins"].astype(float)
            den = (sub.loc[mask, "I_cs_raw"] + eps).astype(float)
            tmp_kt = (num / den).clip(0, 2.0)
            try:
                thresh = tmp_kt.quantile(0.95)
                cand = sub.loc[tmp_kt.index[tmp_kt >= thresh]]
            except Exception:
                cand = sub

        num2 = cand["nins"].astype(float)
        den2 = (cand["I_cs_raw"] + eps).astype(float)
        if len(cand) == 0 or (den2 <= 0).all():
            alpha = 1.0
        else:
            alpha = float(np.median(num2 / den2))
        alphas[pv] = np.clip(alpha, 0.7, 1.3)

    if len(alphas) == 0:
        global_alpha = 1.0
    else:
        global_alpha = float(np.median(list(alphas.values())))
    log(f"[Clear-sky] per-PV alpha (len={len(alphas)}), global_alpha={global_alpha:.3f}")

    data_feat["alpha_pv"] = data_feat["pv_id"].map(alphas).fillna(global_alpha).astype(np.float32)
    data_feat["I_cs"] = (data_feat["I_cs_raw"] * data_feat["alpha_pv"]).astype(np.float32)

    # train 행에만 kt 생성
    data_feat["kt"] = np.nan
    mask_kt_train = is_train & (data_feat["I_cs"] > 0)
    data_feat.loc[mask_kt_train, "kt"] = (
        data_feat.loc[mask_kt_train, "nins"] /
        (data_feat.loc[mask_kt_train, "I_cs"] + eps)
    ).clip(0.0, 1.4).astype(np.float32)

    # 메모리 줄이기
    data_feat = downcast_numeric(data_feat)

    log("[Stage1] 준비 완료: exogenous + kt (train only)")

    # =========================
    # Stage 1: kt (exogenous only, GPU/CPU 공통)
    # =========================

    # 1단계 학습용 데이터 (train 전체 중 kt가 있는 행)
    df_tr1 = data_feat[is_train & data_feat["kt"].notna()].copy()
    df_te1 = data_feat[~is_train].copy()

    # 원래 index 저장 (나중에 kt_hat_stage1 할당용)
    df_tr1["orig_idx"] = df_tr1.index.values
    df_te1["orig_idx"] = df_te1.index.values

    # (1) 낮 + 태양고도 컷으로 행 수 줄이기
    mask_day_elev = (df_tr1["is_day"] == 1) & (df_tr1["solar_elev_deg"] >= ELEV_MIN)
    before_rows = len(df_tr1)
    df_tr1 = df_tr1[mask_day_elev]
    log(f"[Stage1] day & elev cut: {before_rows} -> {len(df_tr1)} rows")

    # (2) 그래도 너무 많으면 샘플링 (메모리 보호용)
    if len(df_tr1) > MAX_STAGE1_ROWS:
        log(f"[Stage1] too many rows ({len(df_tr1)}). "
            f"Random sampling to {MAX_STAGE1_ROWS} rows for Stage1.")
        df_tr1 = df_tr1.sample(MAX_STAGE1_ROWS, random_state=SEED)

    df_tr1 = df_tr1.reset_index(drop=True)

    # feature 선택: nins, kt, is_train, time, pv_id, type, energy, minute 등 제외
    exclude_cols_1 = {
        "time","pv_id","type","energy","nins","kt",
        "is_train","minute"
    }
    feature_candidates_1 = [
        c for c in df_tr1.columns
        if c not in exclude_cols_1 and np.issubdtype(df_tr1[c].dtype, np.number)
    ]

    log(f"[Stage1] candidate features = {len(feature_candidates_1)} "
        f"(e.g. {feature_candidates_1[:12]}{' ...' if len(feature_candidates_1)>12 else ''})")

    X_tr1 = df_tr1[feature_candidates_1].to_numpy(dtype=np.float32, copy=False)
    y_tr1 = df_tr1["kt"].to_numpy(dtype=np.float32, copy=False)
    X_te1 = df_te1[feature_candidates_1].to_numpy(dtype=np.float32, copy=False)

    # NaN row 제거 (안전용)
    keep1 = np.isfinite(X_tr1).all(axis=1) & np.isfinite(y_tr1)
    if not keep1.all():
        log(f"[Stage1] drop {(~keep1).sum()} rows (NaN in features/kt)")
        X_tr1 = X_tr1[keep1]
        y_tr1 = y_tr1[keep1]
        df_tr1 = df_tr1.loc[keep1].reset_index(drop=True)
    else:
        df_tr1 = df_tr1.reset_index(drop=True)

    # GroupKFold by pv_id
    groups = df_tr1["pv_id"].values
    unique_pv = np.unique(groups)
    n_splits = 5 if len(unique_pv) >= 5 else max(2, len(unique_pv))
    gkf = GroupKFold(n_splits=n_splits)

    oof_pred = np.zeros(len(df_tr1), dtype=np.float32)
    test_pred_accum = np.zeros(len(df_te1), dtype=np.float32)

    log(f"[Stage1] GroupKFold splits={n_splits}, train_rows={len(df_tr1)}, test_rows={len(df_te1)}")

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr1, y_tr1, groups), 1):
        log(f"\n[Stage1][Fold {fold}] train={len(tr_idx)}, valid={len(va_idx)}")
        X_tr_f, y_tr_f = X_tr1[tr_idx], y_tr1[tr_idx]
        X_va_f, y_va_f = X_tr1[va_idx], y_tr1[va_idx]

        model1 = LGBMRegressor(
            **LGBM_DEVICE_PARAMS,
            objective="regression_l1",
            random_state=SEED + fold,
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=128,
            min_data_in_leaf=200,
            subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        )
        model1.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(stopping_rounds=300),
                lgb.log_evaluation(period=200)
            ]
        )

        best_iter = getattr(model1, "best_iteration_", None)
        oof_pred[va_idx] = model1.predict(X_va_f, num_iteration=best_iter).astype(np.float32)

        # test 예측 누적
        pred_te_f = model1.predict(X_te1, num_iteration=best_iter).astype(np.float32)
        test_pred_accum += pred_te_f

    # OOF 성능(kt 기준)
    mae_kt_stage1 = mean_absolute_error(y_tr1, oof_pred)
    log(f"\n[Stage1] OOF MAE on kt: {mae_kt_stage1:.6f}")

    kt_hat_stage1_tr = oof_pred
    kt_hat_stage1_te = test_pred_accum / n_splits

    # 원본 data_feat에 kt_hat_stage1 할당
    data_feat["kt_hat_stage1"] = np.nan

    # train
    idx_tr = df_tr1["orig_idx"].values
    data_feat.loc[idx_tr, "kt_hat_stage1"] = kt_hat_stage1_tr

    # test
    idx_te = df_te1["orig_idx"].values
    data_feat.loc[idx_te, "kt_hat_stage1"] = kt_hat_stage1_te

    log("[Stage1] kt_hat_stage1 assigned to all rows")

    # =========================
    # Stage 2: kt using kt_hat_stage1 lag/rolling
    # =========================

    log("\n[Stage2] kt_hat_stage1 lag/rolling 생성")
    data_feat = add_lag_rolling(
        data_feat,
        by="pv_id", time_col="time", target="kt_hat_stage1",
        lag_steps=LAG_STEPS,
        roll_specs=ROLL_SPECS,
        roll_min_periods=ROLL_MIN_PERIODS
    )

    # 다시 정렬
    data_feat = data_feat.sort_values(["pv_id","time"]).reset_index(drop=True)

    # Stage2 학습용 마스크: train + 낮 + 태양고도 컷 + kt 존재
    is_train = data_feat["is_train"] == 1
    mask_tr2 = is_train & (data_feat["is_day"] == 1) & \
               (data_feat["solar_elev_deg"] >= ELEV_MIN) & \
               data_feat["kt"].notna()

    log(f"[Stage2] train rows after day/elev cut: {mask_tr2.sum()}")

    # train/val PV 분리 (홀드아웃)
    train_pvs = sorted(data_feat.loc[mask_tr2, "pv_id"].unique().tolist())
    n_val_pv = max(1, int(len(train_pvs) * 0.12))
    val_pvs = set(random.sample(train_pvs, n_val_pv))

    val_mask = mask_tr2 & data_feat["pv_id"].isin(val_pvs)
    tr_mask = mask_tr2 & ~data_feat["pv_id"].isin(val_pvs)

    log(f"[Stage2] train_pvs={len(train_pvs)}, val_pvs={len(val_pvs)}")
    log(f"[Stage2] rows: train={tr_mask.sum()}, valid={val_mask.sum()}")

    # Stage2 feature 선택
    exclude_cols_2 = {
        "time","pv_id","type","energy",
        "nins","kt","is_train","minute"
    }
    feature_candidates_2 = [
        c for c in data_feat.columns
        if c not in exclude_cols_2 and np.issubdtype(data_feat[c].dtype, np.number)
    ]

    log(f"[Stage2] candidate features = {len(feature_candidates_2)} "
        f"(e.g. {feature_candidates_2[:12]}{' ...' if len(feature_candidates_2)>12 else ''})")

    X_tr2 = data_feat.loc[tr_mask, feature_candidates_2].to_numpy(dtype=np.float32, copy=False)
    y_tr2 = data_feat.loc[tr_mask, "kt"].to_numpy(dtype=np.float32, copy=False)
    X_va2 = data_feat.loc[val_mask, feature_candidates_2].to_numpy(dtype=np.float32, copy=False)
    y_va2 = data_feat.loc[val_mask, "kt"].to_numpy(dtype=np.float32, copy=False)

    # NaN 제거
    keep_tr2 = np.isfinite(X_tr2).all(axis=1) & np.isfinite(y_tr2)
    if not keep_tr2.all():
        log(f"[Stage2] drop {(~keep_tr2).sum()} rows in train (NaN)")
        X_tr2 = X_tr2[keep_tr2]
        y_tr2 = y_tr2[keep_tr2]

    keep_va2 = np.isfinite(X_va2).all(axis=1) & np.isfinite(y_va2)
    if not keep_va2.all():
        log(f"[Stage2] drop {(~keep_va2).sum()} rows in valid (NaN)")
        X_va2 = X_va2[keep_va2]
        y_va2 = y_va2[keep_va2]

    log("[Stage2] LightGBM 학습 시작 (kt target)")
    lgbm2 = LGBMRegressor(
        **LGBM_DEVICE_PARAMS,
        objective="regression_l1",
        random_state=SEED,
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=200,
        subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )
    lgbm2.fit(
        X_tr2, y_tr2,
        eval_set=[(X_va2, y_va2)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=400),
            lgb.log_evaluation(period=200)
        ]
    )

    best_iter2 = getattr(lgbm2, "best_iteration_", None)

    # 검증구간 성능 (nins 기준)
    kt_hat_va = lgbm2.predict(X_va2, num_iteration=best_iter2).astype(np.float32)
    I_cs_va = data_feat.loc[val_mask, "I_cs"].values[:len(kt_hat_va)].astype(np.float32)
    nins_va_true = data_feat.loc[val_mask, "nins"].values[:len(kt_hat_va)].astype(np.float32)
    nins_va_pred = (kt_hat_va * I_cs_va).astype(np.float32)
    is_day_va = data_feat.loc[val_mask, "is_day"].values[:len(kt_hat_va)]
    nins_va_pred[is_day_va == 0] = 0.0
    nins_va_pred = np.where(nins_va_pred < 0, 0, nins_va_pred)

    mae_val_nins = mean_absolute_error(nins_va_true, nins_va_pred)
    log(f"[Stage2] Validation MAE on nins (day/elev cut): {mae_val_nins:.6f}")

    # =========================
    # 최종 test 예측 & 제출 파일 생성
    # =========================

    log("\n[Predict] on official test.csv")

    is_test = data_feat["is_train"] == 0
    X_test2 = data_feat.loc[is_test, feature_candidates_2].to_numpy(dtype=np.float32, copy=False)
    kt_hat_test_final = lgbm2.predict(X_test2, num_iteration=best_iter2).astype(np.float32)

    I_cs_test = data_feat.loc[is_test, "I_cs"].to_numpy(dtype=np.float32, copy=False)
    is_day_test = data_feat.loc[is_test, "is_day"].to_numpy(copy=False)

    pred_nins_test = (kt_hat_test_final * I_cs_test).astype(np.float32)
    pred_nins_test[is_day_test == 0] = 0.0
    pred_nins_test = np.where(pred_nins_test < 0, 0, pred_nins_test)

    # test 부분 원본 순서 유지 위해 index 저장
    test_idx = data_feat.index[is_test]
    sub = data_feat.loc[test_idx, ["time","pv_id"]].copy()
    sub = sub.reset_index(drop=True)
    sub["nins"] = pred_nins_test

    sub.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    log(f"✅ submission saved: {SUBMISSION_PATH} (cols: time, pv_id, nins)")
    log(f"[All Done] total {t.time()-start_all:.2f}s")
    log("===== Two-stage kt pipeline (GPU-ready) end =====")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {e}")
        traceback.print_exc(limit=2)
