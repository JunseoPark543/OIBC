# -*- coding: utf-8 -*-
"""
test_52.146 에서 학습 때 밤의 데이터는 제외시키기
OIBC irradiance pipeline (낮 데이터만 학습 버전)
- 입력: 보간+태양피처 포함 데이터(train_imputed_with_solar.csv)
- 학습: 낮(is_day==1) 데이터만 사용
- 예측: 전체 시간대에 대해 예측 후 밤=0 규칙 적용
- 메모리 최적화: float32 downcast, copy 최소화
- 저장: 예측결과 CSV
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import time as t
import traceback, gc
from pathlib import Path

# =========================
# 0) 경로 설정
# =========================
BASE = Path("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA/test_52.146")
LOAD_IMPUTED_SOLAR = BASE / "train_imputed_with_solar.csv"   # 이미 만들어 둔 파일(태양 피처 포함본)
SAVE_PRED_RESULT   = BASE / "pred_results_daytrain.csv"      # 이번 실행 예측 결과 저장

# =========================
# 유틸: 메모리 다운캐스트
# =========================
def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype(np.int32)
    return df

# (백업) 태양 피처 생성 함수
# - 만약 LOAD_IMPUTED_SOLAR에 is_day 등 태양 피처가 없다면 보충 생성
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

    # 주기 피처
    day_angle = 2*np.pi*((g["hour"]*60 + g["minute"]) / (24*60))
    g["sin_time"] = np.sin(day_angle); g["cos_time"] = np.cos(day_angle)
    year_angle = 2*np.pi*(g["doy"] / 365.25)
    g["sin_doy"] = np.sin(year_angle); g["cos_doy"] = np.cos(year_angle)

    # NOAA 근사(고정 위경도)
    lat = np.deg2rad(KOR_LAT)
    lon_deg = KOR_LON
    local_minutes = (g["hour"]*60 + g["minute"]).astype(float)
    gamma = 2*np.pi*((g["doy"] - 1 + (local_minutes/1440.0)) / 365.0)
    eqt = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                  - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    lon_correction_min = 4.0 * (lon_deg - 135.0)  # KST 기준
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
    return g

# =========================
# 1) Load
# =========================
start = t.time()
data = pd.read_csv(LOAD_IMPUTED_SOLAR)
end = t.time()
print(f"[IO] load: {end-start:.2f}s, shape={data.shape}")
print(f"[IO] columns (head): {list(data.columns)[:12]}{' ...' if len(data.columns)>12 else ''}")

# 안전장치: time 파싱, is_day 미존재시 생성
if "time" not in data.columns:
    raise ValueError("입력 파일에 'time' 컬럼이 필요합니다.")

# time 파싱(이미 문자열일 수 있으므로)
data["time"] = pd.to_datetime(data["time"], errors="coerce")

need_solar = any(c not in data.columns for c in ["hour","minute","doy","sin_time","cos_time","sin_doy","cos_doy","solar_elev_deg","is_day"])
if need_solar:
    print("[Solar] 파일에 태양 피처가 없어 새로 생성합니다.")
    try:
        data = add_time_and_solar_features(data, tz="Asia/Seoul")
    except Exception as e:
        print(f"[ERROR] solar features: {e}")
        traceback.print_exc(limit=2)

# 메모리 다운캐스트
data = downcast_numeric(data)
print("[Mem] after downcast:",
      round(data.memory_usage(deep=True).sum()/1024**2, 1), "MB")

# =========================
# 2) Train/Test split (문제에서 준 pv_id 리스트)
# =========================
if "pv_id" not in data.columns:
    raise ValueError("pv_id 컬럼이 필요합니다.")

test_pv_ids_int = [177,178,179,181,184,185,186,187,188,189,190,191,193,195,196,197,198,199,200,201,202,203,205,206,207,208,209]
test_pv_ids = [f"PV_ID_{i}" for i in test_pv_ids_int]
mask_test = data["pv_id"].isin(test_pv_ids)

print(f"[Split] train rows(raw)={(~mask_test).sum()}, test rows={(mask_test).sum()}")

# =========================
# 3) Feature 선택
# =========================
exclude_cols = ["time", "pv_id", "type", "energy", "nins"]  # 타겟/문자열/시간 제외
feature_candidates = [c for c in data.columns if c not in exclude_cols]
features = [c for c in feature_candidates if np.issubdtype(data[c].dtype, np.number)]
print(f"[Feat] {len(features)} features (e.g., {features[:10]}{' ...' if len(features)>10 else ''})")

# =========================
# 4) 낮만 학습 마스크
# =========================
if "is_day" not in data.columns:
    raise ValueError("is_day 컬럼이 필요합니다. (태양 피처 생성 확인)")

train_mask = (~mask_test) & (data["is_day"] == 1)  # 낮만 학습!
test_mask  = mask_test

print(f"[Split] train(day-only) rows={train_mask.sum()}, test(all times) rows={test_mask.sum()}")

# =========================
# 5) 넘파이 뷰 추출 + 결측 제거
# =========================
X_train = data.loc[train_mask, features].to_numpy(dtype=np.float32, copy=False)
y_train = data.loc[train_mask, "nins"].to_numpy(copy=False)
X_test  = data.loc[test_mask,  features].to_numpy(dtype=np.float32, copy=False)
y_test  = data.loc[test_mask,  "nins"].to_numpy(copy=False)

print(f"[X] train={X_train.shape}, test={X_test.shape}")

# 결측 행 제거(학습만)
train_keep = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
if not train_keep.all():
    print(f"[Clean] drop {(~train_keep).sum()} rows with NaN in train")
    X_train = X_train[train_keep]
    y_train = y_train[train_keep]

# =========================
# 6) 모델 학습/예측/평가
# =========================
print("[Train] LGBMRegressor (day-only) start")
lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)

print("[Predict] start")
raw_pred = lgbm.predict(X_test)

# 밤=0 규칙 적용
is_day_test = data.loc[test_mask, "is_day"].to_numpy(copy=False)
pred = raw_pred.astype(np.float32, copy=False)
pred[is_day_test == 0] = 0.0
pred = np.where(pred < 0, 0, pred)

mae = mean_absolute_error(y_test, pred)
print(f"[Eval] MAE (day-only train + night=0): {mae:.6f}")

# =========================
# 7) 예측 결과 저장
# =========================
result_cols = ["pv_id", "time", "nins"]
result_df = data.loc[test_mask, result_cols].copy()
result_df["pred_nins"] = pred.astype(np.float32)
result_df["abs_error"] = (result_df["nins"] - result_df["pred_nins"]).abs().astype(np.float32)

result_df.to_csv(SAVE_PRED_RESULT, index=False, encoding="utf-8-sig")
print(f"✅ 예측 결과 저장 완료: {SAVE_PRED_RESULT}")
print("컬럼 예시:", ["pv_id", "time", "nins", "pred_nins", "abs_error"])

# =========================
# 8) 메모리 정리
# =========================
del X_train, X_test, y_train, y_test, raw_pred, pred
gc.collect()
