# -*- coding: utf-8 -*-
"""

OIBC irradiance pipeline (Day-only Train + Rowwise & Lag/Rolling Features)
- 입력: 보간+태양피처 포함 데이터(train_imputed_with_solar.csv)
- 전처리:
  (1) 행 단위 파생변수 추가 (물리/대기 프록시 등)
  (2) 시계열 파생변수 추가 (pv_id별 lag/rolling, shift로 누수 방지)
- 학습: 낮(is_day==1) 데이터만 사용 (테스트는 전체)
- 예측: 테스트 전체 시간대 예측 후, 밤=0 규칙 강제
- 저장: pred_results_daytrain_lagroll.csv
- 참고: 대용량 고려하여 downcast 및 copy 최소화
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import time as t
import traceback, gc
from pathlib import Path
import random

# =========================
# 0) 경로/파라미터 설정
# =========================
BASE = Path("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA")
LOAD_IMPUTED_SOLAR = BASE / "train_imputed_with_solar.csv"
SAVE_PRED_RESULT   = BASE / "pred_results_daytrain_lagroll.csv"

# Lag/Rolling 설정 (행 기준; 5분 간격 가정 시 lag12는 1시간 전)
LAG_STEPS   = [1, 2, 3, 6, 12]                 # 바로 이전 ~ 1시간 이전
ROLL_SPECS  = [(3, "mean"), (12, "mean")]      # 15분/1시간 이동평균 (shift(1)로 누수 방지)
# 롤링 최소 관측치 (초반 NaN 방지하려면 1로; 엄격하게 하려면 윈도우와 동일)
ROLL_MIN_PERIODS = 1

# 테스트 PV 목록(문제 제공)import random
numbers = list(range(210))
exclude = [7, 8, 10, 23, 29, 39, 48, 49, 64, 72, 78, 82, 95, 114, 117, 121,
           122, 134, 165, 173, 175, 180, 182, 183, 192, 194, 204]
available = [n for n in numbers if n not in exclude]
TEST_PV_IDS_INT = list(random.sample(available, 27))
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
    # time 파싱 + 타임존 정리
    if not pd.api.types.is_datetime64_any_dtype(g["time"]):
        g["time"] = pd.to_datetime(g["time"], errors="coerce")

    if g["time"].dt.tz is None:
        g["time"] = g["time"].dt.tz_localize(tz)
    else:
        g["time"] = g["time"].dt.tz_convert(tz)

    g["hour"] = g["time"].dt.hour
    g["minute"] = g["time"].dt.minute
    g["doy"] = g["time"].dt.dayofyear

    day_angle = 2*np.pi*((g["hour"]*60 + g["minute"]) / (24*60))
    g["sin_time"] = np.sin(day_angle); g["cos_time"] = np.cos(day_angle)
    year_angle = 2*np.pi*(g["doy"] / 365.25)
    g["sin_doy"] = np.sin(year_angle); g["cos_doy"] = np.cos(year_angle)

    lat = np.deg2rad(KOR_LAT)
    lon_deg = KOR_LON
    local_minutes = (g["hour"]*60 + g["minute"]).astype(float)
    gamma = 2*np.pi*((g["doy"] - 1 + (local_minutes/1440.0)) / 365.0)
    eqt = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                  - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    lon_correction_min = 4.0 * (lon_deg - 135.0)  # KST 기준 중앙경도
    time_offset = eqt + lon_correction_min
    tst = local_minutes + time_offset
    hra = np.deg2rad((tst/4.0) - 180.0)
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))
    cos_zenith = np.sin(np.deg2rad(90.0 - (90.0 - np.rad2deg(np.arccos(
        np.clip(np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(hra), -1.0, 1.0),)))
    ))  # 안전용 꼼수지만 아래서 다시 정확히 계산
    # 정확 계산
    cos_zenith = np.sin(lat)*np.sin(decl) + np.cos(lat)*np.cos(decl)*np.cos(hra)
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.arccos(cos_zenith)
    solar_elev_deg = 90.0 - np.rad2deg(zenith)

    g["solar_elev_deg"] = solar_elev_deg.astype(np.float32)
    g["is_day"] = (solar_elev_deg > 0.0).astype(np.int8)

    # 중간값 노출(선택): eqt/hra/decl
    g["eqt"] = eqt.astype(np.float32)
    g["hra"] = ((tst/4.0) - 180.0).astype(np.float32)            # (deg)
    g["decl"] = np.rad2deg(decl).astype(np.float32)              # (deg)
    return g

# =========================
# 행 단위 파생변수 (누수 없음)
# =========================
def add_rowwise_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()

    # 안전 캐스팅
    for c in ["solar_elev_deg", "temp_a", "dew_point", "vis", "real_feel_temp",
              "real_feel_temp_shade", "appr_temp", "wind_chill_temp",
              "ground_press", "pressure", "uv_idx", "rain", "snow", "precip_1h"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    # 1) 태양기하 파생
    if "solar_elev_deg" in g.columns:
        elev_rad = np.deg2rad(g["solar_elev_deg"].astype(float))
        # cos(zenith) = sin(elev)
        cosZ = np.sin(elev_rad)
        g["cosZ_pos"] = np.clip(cosZ, 0.0, None).astype(np.float32)
        g["sin_elev"] = np.sin(np.clip(elev_rad, 0, None)).astype(np.float32)
        g["elev_sq"] = (g["solar_elev_deg"].clip(lower=0)**2).astype(np.float32)

    # decl/hra/eqt가 이미 있다면 그대로 활용(없어도 무방)
    for c in ["decl", "hra", "eqt"]:
        if c in g.columns:
            g[c] = g[c].astype(np.float32)

    # 2) 대기/혼탁 프록시
    if {"temp_a","dew_point"} <= set(g.columns):
        g["dewpoint_depr"] = (g["temp_a"] - g["dew_point"]).astype(np.float32)

    if "vis" in g.columns:
        g["vis_inv"] = (1.0 / (g["vis"].abs() + 1e-3)).astype(np.float32)
        g["vis_log"] = np.log1p(g["vis"].clip(lower=0)).astype(np.float32)

    if {"ground_press","pressure"} <= set(g.columns):
        g["press_diff"] = (g["ground_press"] - g["pressure"]).astype(np.float32)

    # 3) 체감온도 차이
    if {"real_feel_temp","temp_a"} <= set(g.columns):
        g["realfeel_gap"] = (g["real_feel_temp"] - g["temp_a"]).astype(np.float32)
    if {"real_feel_temp_shade","temp_a"} <= set(g.columns):
        g["shade_gap"] = (g["real_feel_temp_shade"] - g["temp_a"]).astype(np.float32)
    if {"appr_temp","temp_a"} <= set(g.columns):
        g["appr_gap"] = (g["appr_temp"] - g["temp_a"]).astype(np.float32)
    if {"wind_chill_temp","temp_a"} <= set(g.columns):
        g["windchill_gap"] = (g["wind_chill_temp"] - g["temp_a"]).astype(np.float32)

    # 4) 강수 여부/형태
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

    # 5) UV 정규화
    if {"uv_idx","cosZ_pos"} <= set(g.columns):
        g["uv_norm"] = (g["uv_idx"] / (g["cosZ_pos"] + 1e-3)).astype(np.float32)

    return g

# =========================
# 시계열 파생변수 (누수 방지: shift)
# =========================
def add_lag_rolling(
    df: pd.DataFrame,
    by="pv_id",
    time_col="time",
    target="nins",
    lag_steps=(1,2,3,6,12),
    roll_specs=((3,"mean"), (12,"mean")),
    roll_min_periods=1
) -> pd.DataFrame:
    g = df.sort_values([by, time_col]).copy()

    # Lag
    for L in lag_steps:
        col = f"{target}_lag{L}"
        g[col] = g.groupby(by, sort=False)[target].shift(L)

    # Rolling (shift(1)로 현재시점 누수 방지)
    for win, agg in roll_specs:
        base = g.groupby(by, sort=False)[target].shift(1)
        rolled = base.groupby(g[by], sort=False).rolling(window=win, min_periods=roll_min_periods).agg(agg)
        # rolling 후 인덱스가 MultiIndex일 수 있음 → 값만 재배치
        g[f"{target}_roll{win}_{agg}"] = rolled.reset_index(level=0, drop=True)

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
    if "time" not in data.columns or "pv_id" not in data.columns or "nins" not in data.columns:
        raise ValueError("필수 컬럼(time, pv_id, nins)이 누락되었습니다.")

    # 2) time 파싱 (tz는 불문, 정렬만 필요)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")

    # 3) 태양 피처 보완(없으면 생성)
    need_solar = any(c not in data.columns for c in
                     ["hour","minute","doy","sin_time","cos_time","sin_doy","cos_doy","solar_elev_deg","is_day"])
    if need_solar:
        print("[Solar] 태양 피처가 없어 새로 생성합니다.")
        data = add_time_and_solar_features(data, tz="Asia/Seoul")

    # 4) 다운캐스트
    data = downcast_numeric(data)
    print("[Mem] after downcast:",
          round(data.memory_usage(deep=True).sum()/1024**2, 1), "MB")

    # 5) 행 단위 파생
    data = add_rowwise_engineered_features(data)

    # 6) Lag/Rolling (pv별, 시간 정렬, 누수 방지)
    data = add_lag_rolling(
        data,
        by="pv_id", time_col="time", target="nins",
        lag_steps=LAG_STEPS, roll_specs=ROLL_SPECS, roll_min_periods=ROLL_MIN_PERIODS
    )

    # 7) Train/Test split (PV 단위 홀드아웃)
    mask_test = data["pv_id"].isin(TEST_PV_IDS)
    print(f"[Split] train raw rows={(~mask_test).sum()}, test rows={mask_test.sum()}")

    # 8) Feature 선택
    #    - is_day: 학습 데이터는 전부 1 → 정보 없음 → 제외
    #    - minute: sin/cos_time가 있어 중복 고주파 → 우선 제외
    exclude_cols = ["time", "pv_id", "type", "energy", "nins", "is_day", "minute"]
    feature_candidates = [c for c in data.columns if c not in exclude_cols]
    features = [c for c in feature_candidates if np.issubdtype(data[c].dtype, np.number)]

    print(f"[Feat] {len(features)} features (e.g., {features[:12]}{' ...' if len(features)>12 else ''})")

    # 9) 낮만 학습 마스크
    train_mask = (~mask_test) & (data["is_day"] == 1)
    test_mask  = mask_test

    # 10) 넘파이 뷰 추출
    X_train = data.loc[train_mask, features].to_numpy(dtype=np.float32, copy=False)
    y_train = data.loc[train_mask, "nins"].to_numpy(copy=False)
    X_test  = data.loc[test_mask,  features].to_numpy(dtype=np.float32, copy=False)
    y_test  = data.loc[test_mask,  "nins"].to_numpy(copy=False)

    print(f"[X] train={X_train.shape}, test={X_test.shape}")

    # 11) 결측 제거(학습만)
    train_keep = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    if not train_keep.all():
        print(f"[Clean] drop {(~train_keep).sum()} rows with NaN in train (due to early lags/rollings)")
        X_train = X_train[train_keep]
        y_train = y_train[train_keep]

    # 12) 모델 학습
    print("[Train] LGBMRegressor (day-only + lag/rolling) start")
    lgbm = LGBMRegressor(
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1
    )
    lgbm.fit(X_train, y_train)

    # 13) 예측 + 밤=0 규칙
    print("[Predict] start")
    raw_pred = lgbm.predict(X_test)
    is_day_test = data.loc[test_mask, "is_day"].to_numpy(copy=False)
    pred = raw_pred.astype(np.float32, copy=False)
    pred[is_day_test == 0] = 0.0
    pred = np.where(pred < 0, 0, pred)

    # 14) 평가
    mae = mean_absolute_error(y_test, pred)
    print(f"[Eval] MAE (day-only train + night=0 + lag/rolling): {mae:.6f}")

    # 15) 예측 결과 저장
    result_cols = ["pv_id", "time", "nins"]
    result_df = data.loc[test_mask, result_cols].copy()
    result_df["pred_nins"] = pred.astype(np.float32)
    result_df["abs_error"] = (result_df["nins"] - result_df["pred_nins"]).abs().astype(np.float32)
    result_df.to_csv(SAVE_PRED_RESULT, index=False, encoding="utf-8-sig")
    print(f"✅ 예측 결과 저장 완료: {SAVE_PRED_RESULT}")
    print("컬럼 예시:", ["pv_id", "time", "nins", "pred_nins", "abs_error"])

    # 16) Feature Importance (상위 30개)
    try:
        importances = lgbm.feature_importances_
        fi = pd.DataFrame({"feature": features, "importance": importances})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        print("\n[Top-30 Feature Importances]")
        for i, row in fi.head(30).iterrows():
            print(f"{i+1:2d}. {row['feature']}: {int(row['importance'])}")
    except Exception as e:
        print(f"[WARN] feature importance 표시 실패: {e}")

    # 17) 메모리 정리
    del X_train, X_test, y_train, y_test
    del raw_pred, pred
    gc.collect()

    print(f"\n[All Done] total {t.time()-start_all:.2f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc(limit=2)
