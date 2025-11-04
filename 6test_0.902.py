# -*- coding: utf-8 -*-
"""
OIBC irradiance pipeline — FULL (kt target + clear-sky + rowwise + dynamics + lag/rolling)
- 입력: 보간+태양피처 포함(train_imputed_with_solar.csv)
- 타깃: kt = nins / I_cs  (Haurwitz clear-sky)
- 전처리:
  (1) 행 단위 파생 (물리/대기 프록시)
  (2) 날씨 변화율/롤링 (누수 방지: shift)
  (3) nins lag/rolling (누수 방지: shift)
  (4) clear-sky I_cs 및 kt 생성
- 학습: 낮(is_day==1) & 태양고도 >= ELEV_MIN  (테스트는 전체)
- 모델: LightGBM, objective=regression_l1(MAE), early stopping(검증=훈련 PV 중 일부)
- 후처리: 밤=0 규칙, 음수 컷, nins 복원(kt_hat * I_cs)
- 저장: pred_results_kt_daytrain_lagroll_clearsky.csv
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb   # ★ 콜백용
from sklearn.metrics import mean_absolute_error
import time as t
import traceback, gc, random
from pathlib import Path

# =========================
# 0) 경로/파라미터 설정
# =========================
BASE = Path("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA")
LOAD_IMPUTED_SOLAR = BASE / "train_imputed_with_solar.csv"
SAVE_PRED_RESULT   = BASE / "pred_results_kt_daytrain_lagroll_clearsky.csv"

# 재현성
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Lag/Rolling 설정 (5분 간격 가정)
LAG_STEPS   = [1, 2, 3, 6, 12]               # 5~60분 전
ROLL_SPECS  = [(3, "mean"), (12, "mean")]    # 15분/60분 평균
ROLL_MIN_PERIODS = 1

# 훈련 시 해뜰/해질 구간 제외 임계 (°)
ELEV_MIN = 5.0

# 테스트 PV 목록 (대회 규정 리스트가 있다면 고정 사용 권장)
numbers = list(range(210))
exclude = [7, 8, 10, 23, 29, 39, 48, 49, 64, 72, 78, 82, 95, 114, 117, 121,
           122, 134, 165, 173, 175, 180, 182, 183, 192, 194, 204]
available = [n for n in numbers if n not in exclude]
TEST_PV_IDS_INT = sorted(random.sample(available, 27))
# 고정하려면 아래 라인 사용:
# TEST_PV_IDS_INT = [177,178,179,181,184,185,186,187,188,189,190,191,193,195,196,197,198,199,200,201,202,203,205,206,207,208,209]
TEST_PV_IDS = [f"PV_ID_{i}" for i in TEST_PV_IDS_INT]

# =========================
# 유틸: 메모리 다운캐스트
# =========================
def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype(np.int32)
    return df

# (백업) 태양 피처 생성 함수 (파일에 없을 때만 사용)
KOR_LAT = 36.5
KOR_LON = 127.9

def add_time_and_solar_features(df: pd.DataFrame, tz="Asia/Seoul") -> pd.DataFrame:
    g = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(g["time"]):
        g["time"] = pd.to_datetime(g["time"], errors="coerce")
    if g["time"].dt.tz is None:
        g["time"] = g["time"].dt.tz_localize(tz)
    else:
        g["time"] = g["time"].dt.tz_convert(tz)

    g["hour"]   = g["time"].dt.hour
    g["minute"] = g["time"].dt.minute
    g["doy"]    = g["time"].dt.dayofyear

    day_angle  = 2*np.pi*((g["hour"]*60 + g["minute"]) / (24*60))
    g["sin_time"] = np.sin(day_angle); g["cos_time"] = np.cos(day_angle)
    year_angle = 2*np.pi*(g["doy"] / 365.25)
    g["sin_doy"]  = np.sin(year_angle); g["cos_doy"] = np.cos(year_angle)

    lat = np.deg2rad(KOR_LAT); lon_deg = KOR_LON
    local_minutes = (g["hour"]*60 + g["minute"]).astype(float)
    gamma = 2*np.pi*((g["doy"] - 1 + (local_minutes/1440.0)) / 365.0)
    eqt = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                  - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    lon_correction_min = 4.0 * (lon_deg - 135.0)  # 한국 표준시 기준
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

    # 중간값(참고용)
    g["eqt"]  = eqt.astype(np.float32)
    g["hra"]  = ((tst/4.0) - 180.0).astype(np.float32) # deg
    g["decl"] = np.rad2deg(decl).astype(np.float32)    # deg
    return g

# =========================
# (1) 행 단위 파생 (누수 없음)
# =========================
def add_rowwise_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    for c in ["solar_elev_deg", "temp_a", "dew_point", "vis", "real_feel_temp",
              "real_feel_temp_shade", "appr_temp", "wind_chill_temp",
              "ground_press", "pressure", "uv_idx", "rain", "snow", "precip_1h"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    # 태양기하 파생
    if "solar_elev_deg" in g.columns:
        elev_rad = np.deg2rad(g["solar_elev_deg"].astype(float))
        cosZ = np.sin(elev_rad)  # cos(zenith) = sin(elev)
        g["cosZ_pos"] = np.clip(cosZ, 0.0, None).astype(np.float32)
        g["sin_elev"] = np.sin(np.clip(elev_rad, 0, None)).astype(np.float32)
        g["elev_sq"]  = (g["solar_elev_deg"].clip(lower=0)**2).astype(np.float32)

    # decl/hra/eqt 캐스팅
    for c in ["decl","hra","eqt"]:
        if c in g.columns:
            g[c] = g[c].astype(np.float32)

    # 대기/혼탁 프록시
    if {"temp_a","dew_point"} <= set(g.columns):
        g["dewpoint_depr"] = (g["temp_a"] - g["dew_point"]).astype(np.float32)

    if "vis" in g.columns:
        g["vis_inv"] = (1.0 / (g["vis"].abs() + 1e-3)).astype(np.float32)
        g["vis_log"] = np.log1p(g["vis"].clip(lower=0)).astype(np.float32)

    if {"ground_press","pressure"} <= set(g.columns):
        g["press_diff"] = (g["ground_press"] - g["pressure"]).astype(np.float32)

    # 체감온도 차이
    if {"real_feel_temp","temp_a"} <= set(g.columns):
        g["realfeel_gap"] = (g["real_feel_temp"] - g["temp_a"]).astype(np.float32)
    if {"real_feel_temp_shade","temp_a"} <= set(g.columns):
        g["shade_gap"] = (g["real_feel_temp_shade"] - g["temp_a"]).astype(np.float32)
    if {"appr_temp","temp_a"} <= set(g.columns):
        g["appr_gap"] = (g["appr_temp"] - g["temp_a"]).astype(np.float32)
    if {"wind_chill_temp","temp_a"} <= set(g.columns):
        g["windchill_gap"] = (g["wind_chill_temp"] - g["temp_a"]).astype(np.float32)

    # 강수 여부/형태
    rain = g["rain"] if "rain" in g.columns else 0
    snow = g["snow"] if "snow" in g.columns else 0
    precip = g["precip_1h"] if "precip_1h" in g.columns else 0
    g["precip_flag"] = ((rain>0) | (snow>0) | (precip>0)).astype(np.int8)

    wet_code = np.zeros(len(g), dtype=np.int8)
    if "rain" in g.columns: wet_code = np.where(g["rain"]>0, 1, wet_code)
    if "snow" in g.columns: wet_code = np.where(g["snow"]>0, 2, wet_code)
    g["wet_code"] = wet_code

    # UV 정규화
    if {"uv_idx","cosZ_pos"} <= set(g.columns):
        g["uv_norm"] = (g["uv_idx"] / (g["cosZ_pos"] + 1e-3)).astype(np.float32)
    return g

# =========================
# (2) 날씨 변화율/롤링 (누수 방지: shift)
# =========================
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

# =========================
# (3) nins lag/rolling (누수 방지: shift)
# =========================
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
# (4) Clear-sky 계산 및 kt 생성
# =========================
def compute_clear_sky_haurwitz(df, elev_col="solar_elev_deg"):
    cs = np.zeros(len(df), dtype=np.float32)
    if elev_col in df.columns:
        elev_rad = np.deg2rad(df[elev_col].astype(float))
        cosZ = np.clip(np.sin(elev_rad), 0.0, None)  # cos(zenith) = sin(elev)
        mask = cosZ > 0
        tmp = np.zeros_like(cosZ, dtype=np.float64)
        tmp[mask] = 1098.0 * cosZ[mask] * np.exp(-0.057 / np.clip(cosZ[mask], 1e-6, None))
        cs = tmp.astype(np.float32)
    return cs

def add_clear_sky_and_kt(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["I_cs"] = compute_clear_sky_haurwitz(g, "solar_elev_deg")
    eps = 1e-3
    g["kt"] = (g["nins"] / (g["I_cs"] + eps)).astype(np.float32)
    g["kt"] = g["kt"].clip(lower=0.0, upper=2.0)  # 안정화
    return g

# =========================
# 메인 파이프라인
# =========================
def main():
    start_all = t.time()

    # 1) Load
    t0 = t.time()
    data = pd.read_csv(LOAD_IMPUTED_SOLAR)
    print(f"[IO] load: {t.time()-t0:.2f}s, shape={data.shape}")
    req = {"time","pv_id","nins"}
    if not req.issubset(data.columns):
        raise ValueError(f"필수 컬럼 누락: {req - set(data.columns)}")

    # 2) time 파싱
    data["time"] = pd.to_datetime(data["time"], errors="coerce")

    # 3) 태양 피처 보완(없으면 생성)
    need_solar = any(c not in data.columns for c in
                     ["hour","minute","doy","sin_time","cos_time","sin_doy","cos_doy","solar_elev_deg","is_day"])
    if need_solar:
        print("[Solar] 태양 피처 생성")
        data = add_time_and_solar_features(data, tz="Asia/Seoul")

    # 4) 다운캐스트
    data = downcast_numeric(data)
    print("[Mem] after downcast:",
          round(data.memory_usage(deep=True).sum()/1024**2, 1), "MB")

    # 5) 행 단위 파생
    data = add_rowwise_engineered_features(data)

    # 6) 날씨 변화율/롤링
    data = add_weather_dynamics(data)

    # 7) nins lag/rolling
    data = add_lag_rolling(
        data,
        by="pv_id", time_col="time", target="nins",
        lag_steps=LAG_STEPS, roll_specs=ROLL_SPECS, roll_min_periods=ROLL_MIN_PERIODS
    )

    # 8) Clear-sky & kt
    data = add_clear_sky_and_kt(data)

    # 9) Train/Test split (PV 단위 홀드아웃)
    mask_test = data["pv_id"].isin(TEST_PV_IDS)
    print(f"[Split] train raw rows={(~mask_test).sum()}, test rows={mask_test.sum()}")

    # 10) Feature 선택
    exclude_cols = ["time","pv_id","type","energy","nins","is_day","minute"]
    feature_candidates = [c for c in data.columns if c not in exclude_cols]
    features = [c for c in feature_candidates if np.issubdtype(data[c].dtype, np.number)]
    print(f"[Feat] {len(features)} features (e.g., {features[:12]}{' ...' if len(features)>12 else ''})")

    # 11) 학습/검증/테스트 마스크 (낮 + 고도 필터)
    train_mask_base = (~mask_test) & (data["is_day"] == 1) & (data["solar_elev_deg"] >= ELEV_MIN)
    test_mask        = mask_test

    # 검증: 훈련 PV 중 일부를 통째로 홀드아웃
    train_pvs = sorted(data.loc[train_mask_base, "pv_id"].unique().tolist())
    n_val_pv  = max(1, int(len(train_pvs) * 0.12))
    val_pvs   = set(random.sample(train_pvs, n_val_pv))
    val_mask  = train_mask_base & data["pv_id"].isin(val_pvs)
    train_mask= train_mask_base & ~data["pv_id"].isin(val_pvs)

    print(f"[Split] train rows={train_mask.sum()}, valid rows={val_mask.sum()}, test rows={test_mask.sum()}")

    # 12) 넘파이 뷰 추출
    X_tr = data.loc[train_mask, features].to_numpy(dtype=np.float32, copy=False)
    y_tr = data.loc[train_mask, "kt"].to_numpy(copy=False)
    X_va = data.loc[val_mask,   features].to_numpy(dtype=np.float32, copy=False)
    y_va = data.loc[val_mask,   "kt"].to_numpy(copy=False)
    X_te = data.loc[test_mask,  features].to_numpy(dtype=np.float32, copy=False)
    y_te = data.loc[test_mask,  "nins"].to_numpy(copy=False)  # 평가는 원 타깃

    # 13) 학습결측 제거 (초기 lag/roll NaN)
    keep_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    if not keep_tr.all():
        print(f"[Clean] drop {(~keep_tr).sum()} rows in train (NaN by lags/rollings)")
        X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr]
    keep_va = np.isfinite(X_va).all(axis=1) & np.isfinite(y_va)
    if not keep_va.all():
        print(f"[Clean] drop {(~keep_va).sum()} rows in valid (NaN by lags/rollings)")
        X_va, y_va = X_va[keep_va], y_va[keep_va]

    # 14) 모델 학습 (MAE 목적 + Early stopping) — ★ 콜백 사용
    print("[Train] LightGBM (kt target)")
    lgbm = LGBMRegressor(
        objective="regression_l1",  # MAE
        random_state=SEED,
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=128,
        min_data_in_leaf=200,
        subsample=0.9,
        subsample_freq=1,
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
            lgb.log_evaluation(period=200)  # 200스텝마다 로그
        ]
    )

    # 15) 예측: kt_hat -> nins_hat 복원 + 밤=0 규칙
    print("[Predict] on test")
    best_iter = getattr(lgbm, "best_iteration_", None)
    kt_hat = lgbm.predict(X_te, num_iteration=best_iter)
    I_cs_te = data.loc[test_mask, "I_cs"].to_numpy(copy=False).astype(np.float32)
    pred = (kt_hat.astype(np.float32) * I_cs_te).astype(np.float32)

    is_day_test = data.loc[test_mask, "is_day"].to_numpy(copy=False)
    pred[is_day_test == 0] = 0.0
    pred = np.where(pred < 0, 0, pred)

    # 16) 평가
    mae = mean_absolute_error(y_te, pred)
    print(f"[Eval] MAE (kt-model + clear-sky + night=0): {mae:.6f}")

    # 17) 저장
    result_cols = ["pv_id","time","nins"]
    result_df = data.loc[test_mask, result_cols].copy()
    result_df["pred_nins"] = pred.astype(np.float32)
    result_df["abs_error"] = (data.loc[test_mask, "nins"].to_numpy(copy=False).astype(np.float32) - result_df["pred_nins"]).abs()
    result_df.to_csv(SAVE_PRED_RESULT, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {SAVE_PRED_RESULT}")
    print("컬럼 예시:", ["pv_id","time","nins","pred_nins","abs_error"])

    # 18) Feature importance
    try:
        importances = lgbm.feature_importances_
        fi = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
            "importance", ascending=False).reset_index(drop=True)
        print("\n[Top-30 Feature Importances]")
        for i, row in fi.head(30).iterrows():
            print(f"{i+1:2d}. {row['feature']}: {int(row['importance'])}")
    except Exception as e:
        print(f"[WARN] feature importance 표시 실패: {e}")

    # 19) 메모리 정리
    del X_tr, X_va, X_te, y_tr, y_va, y_te, kt_hat, pred
    gc.collect()
    print(f"\n[All Done] total {t.time()-start_all:.2f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc(limit=2)
