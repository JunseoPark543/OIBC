# -*- coding: utf-8 -*-
"""
OIBC irradiance pipeline (메모리 안전 + 디버깅)
- 보간 규칙:
  1) Cubic spline: temp_a, temp_b, appr_temp, real_feel_temp, real_feel_temp_shade,
                   wind_chill_temp, temp_max, temp_min, dew_point, rel_hum,
                   humidity, pressure, ground_press, vis, uv_idx
  2) 계단/앞방향 보간: precip_1h, rain, snow
  3) 기타 수치: 선형 보간
- 태양 위치 근사(한국 위경도) + 밤=0 강제
- 메모리 최적화: float32 downcast, copy 최소화, 넘파이 뷰 학습
- 저장: 보간본 / 보간+태양피처 / 예측결과
** 학습에 밤 데이터까지 포함시킴 
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
BASE = Path("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA")
CSV_PATH = BASE / "train.csv"
SAVE_IMPUTED_ONLY   = BASE / "train_imputed_only.csv"
SAVE_IMPUTED_SOLAR  = BASE / "train_imputed_with_solar.csv"
SAVE_PRED_RESULT    = BASE / "pred_results.csv"

# =========================
# 유틸: 메모리 다운캐스트
# =========================
def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        # 음수가 없다면 uint32 고려 가능. 여기선 안전하게 int32
        df[c] = df[c].astype(np.int32)
    return df

# =========================
# 1) Load
# =========================
start = t.time()
data = pd.read_csv(CSV_PATH)
end = t.time()
print(f"[IO] load: {end-start:.2f}s, shape={data.shape}")
print(f"[IO] columns={list(data.columns)}")

# =========================
# 2) Datetime
# =========================
start = t.time()
data["time"] = pd.to_datetime(data["time"], errors="coerce")
n_time_na = data["time"].isna().sum()
end = t.time()
print(f"[Time] parsed in {end-start:.2f}s, time_na={n_time_na}")

# =========================
# 3) 보간 규칙 정의
# =========================
cubic_cols = [
    "temp_a", "temp_b", "appr_temp", "real_feel_temp", "real_feel_temp_shade",
    "wind_chill_temp", "temp_max", "temp_min", "dew_point", "rel_hum",
    "humidity", "pressure", "ground_press", "vis", "uv_idx"
]
step_ffill_cols = ["precip_1h", "rain", "snow"]
exclude_from_interp = {"time", "pv_id", "type", "energy", "nins"}  # 보간 제외

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def impute_by_rules(group_df: pd.DataFrame, pv_id: str) -> pd.DataFrame:
    """pv_id 그룹별 시간 정렬 후 보간 적용 (디버깅 출력 포함)"""
    print(f"\n▶ [Group] {pv_id} rows={len(group_df)}")
    g = group_df.sort_values("time")  # copy() 생략해 메모리 절약

    present_cubic = [c for c in cubic_cols if c in g.columns]
    present_step  = [c for c in step_ffill_cols if c in g.columns]
    g = _coerce_numeric(g, present_cubic + present_step)

    # --- 3차 스플라인 ---
    if present_cubic:
        try:
            print(f"  - cubic_spline: {present_cubic[:6]}{' ...' if len(present_cubic)>6 else ''}")
            g.loc[:, present_cubic] = g.loc[:, present_cubic].interpolate(
                method="spline", order=3, limit_direction="both"
            )
        except Exception as e:
            print(f"  [WARN] spline fail → polynomial fallback: {e}")
            traceback.print_exc(limit=1)
            g.loc[:, present_cubic] = g.loc[:, present_cubic].interpolate(
                method="polynomial", order=3, limit_direction="both"
            )

    # --- 계단(앞방향 보간) ---
    if present_step:
        print(f"  - step_ffill: {present_step}")
        g.loc[:, present_step] = g.loc[:, present_step].ffill()
        # 초기 구간도 0으로 채우려면 아래 주석 해제
        # g.loc[:, present_step] = g.loc[:, present_step].fillna(0)

    # --- 나머지 수치 선형 보간 ---
    numeric_cols = g.select_dtypes(include=[np.number]).columns.tolist()
    other_linear_cols = [
        c for c in numeric_cols
        if c not in present_cubic and c not in present_step and c not in exclude_from_interp
    ]
    if other_linear_cols:
        print(f"  - linear ({len(other_linear_cols)}): {other_linear_cols[:6]}{' ...' if len(other_linear_cols)>6 else ''}")
        try:
            g.loc[:, other_linear_cols] = g.loc[:, other_linear_cols].interpolate(
                method="linear", limit_direction="both"
            )
        except Exception as e:
            print(f"  [ERROR] linear interp: {e}")
            traceback.print_exc(limit=1)

    print(f"  ✓ done: {pv_id}")
    return g

# =========================
# 4) 전체 데이터 보간 (pv_id별)
# =========================
if "pv_id" not in data.columns:
    raise ValueError("pv_id 컬럼이 필요합니다.")

start = t.time()
imputed_list = []
pvs = data["pv_id"].unique().tolist()
print(f"[Impute] groups={len(pvs)}")
for i, pv in enumerate(pvs, 1):
    try:
        imputed_list.append(impute_by_rules(data[data["pv_id"] == pv], pv))
    except Exception as e:
        print(f"[ERROR] impute group {pv}: {e}")
        traceback.print_exc(limit=1)
    if i % 5 == 0 or i == len(pvs):
        print(f"  -> progress {i}/{len(pvs)} ({i/len(pvs)*100:.1f}%)")
data_imputed = pd.concat(imputed_list, ignore_index=True)
end = t.time()
print(f"[Impute] done in {end-start:.2f}s, shape={data_imputed.shape}")

# 보간 직후 float64 → float32 다운캐스트 (메모리 절감)
data_imputed = downcast_numeric(data_imputed)
print("[Mem] after impute+downcast:",
      round(data_imputed.memory_usage(deep=True).sum()/1024**2, 1), "MB")

# (선택) 보간본 저장
data_imputed.to_csv(SAVE_IMPUTED_ONLY, index=False, encoding="utf-8-sig")
print(f"✅ 보간 완료 데이터 저장: {SAVE_IMPUTED_ONLY}")

# =========================
# 5) 태양 위치 근사 + 밤=0 규칙용 피처
# =========================
KOR_LAT = 36.5  # 대략적
KOR_LON = 127.9 # 대략적 (UTC+9 표준자오선 135E와의 차이 보정에 사용)

def add_time_and_solar_features(df: pd.DataFrame, tz="Asia/Seoul") -> pd.DataFrame:
    g = df.copy()
    # tz 처리
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
    # 타임존 중앙경도(한국): 135E
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
    return g

def apply_night_zero_rule(pred: np.ndarray, feat_df: pd.DataFrame) -> np.ndarray:
    out = pred.copy()
    if "is_day" in feat_df.columns:
        night_mask = (feat_df["is_day"].values == 0)
        out[night_mask] = 0.0
    return np.where(out < 0, 0, out)

start = t.time()
try:
    data_feat = add_time_and_solar_features(data_imputed, tz="Asia/Seoul")
    print("[Solar] added features: ['hour','minute','doy','sin_time','cos_time','sin_doy','cos_doy','solar_elev_deg','is_day']")
except Exception as e:
    print(f"[ERROR] solar features: {e}")
    traceback.print_exc(limit=2)
    data_feat = data_imputed
end = t.time()
print(f"[Solar] done in {end-start:.2f}s")

# (선택) 보간+태양피처 저장
data_feat.to_csv(SAVE_IMPUTED_SOLAR, index=False, encoding="utf-8-sig")
print(f"✅ 태양 피처 포함 데이터 저장: {SAVE_IMPUTED_SOLAR}")

# =========================
# 6) Train/Test split (문제에서 준 pv_id 리스트)
#    메모리 절약: mask만 만들고 큰 DataFrame 복사하지 않음
# =========================
test_pv_ids_int = [177,178,179,181,184,185,186,187,188,189,190,191,193,195,196,197,198,199,200,201,202,203,205,206,207,208,209]
test_pv_ids = [f"PV_ID_{i}" for i in test_pv_ids_int]
mask_test = data_feat["pv_id"].isin(test_pv_ids)

print(f"[Split] train rows={(~mask_test).sum()}, test rows={(mask_test).sum()}")

# =========================
# 7) Feature 선택 (수치형만)
# =========================
exclude_cols = ["time", "pv_id", "type", "energy", "nins"]
feature_candidates = [c for c in data_feat.columns if c not in exclude_cols]
features = [c for c in feature_candidates
            if np.issubdtype(data_feat[c].dtype, np.number)]
print(f"[Feat] {len(features)} features (e.g., {features[:10]}{' ...' if len(features)>10 else ''})")

# =========================
# 8) 넘파이 뷰로 학습 데이터 추출 (copy=False)
# =========================
X_train = data_feat.loc[~mask_test, features].to_numpy(dtype=np.float32, copy=False)
y_train = data_feat.loc[~mask_test, "nins"].to_numpy(copy=False)
X_test  = data_feat.loc[ mask_test, features].to_numpy(dtype=np.float32, copy=False)
y_test  = data_feat.loc[ mask_test, "nins"].to_numpy(copy=False)

print(f"[X] train={X_train.shape}, test={X_test.shape}")

# 결측이 남아있다면(드물지만) 행 제거
train_keep = np.isfinite(X_train).all(axis=1)
if not train_keep.all():
    print(f"[Clean] drop {(~train_keep).sum()} rows with NaN in train")
    X_train = X_train[train_keep]
    y_train = y_train[train_keep]

# =========================
# 9) 모델 학습/예측/평가
# =========================
print("[Train] LGBMRegressor start")
lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)

print("[Predict] start")
raw_pred = lgbm.predict(X_test)
# 밤=0 규칙
is_day_test = data_feat.loc[mask_test, "is_day"].to_numpy(copy=False)
pred = raw_pred.copy()
pred[is_day_test == 0] = 0.0
pred = np.where(pred < 0, 0, pred)

mae = mean_absolute_error(y_test, pred)
print(f"[Eval] MAE (night=0 rule): {mae:.6f}")

# =========================
# 10) 예측 결과 저장(필요 컬럼만)
# =========================
result_cols = ["pv_id", "time", "nins"]
result_df = data_feat.loc[mask_test, result_cols].copy()
result_df["pred_nins"] = pred.astype(np.float32)
result_df["abs_error"] = (result_df["nins"] - result_df["pred_nins"]).abs()
result_df.to_csv(SAVE_PRED_RESULT, index=False, encoding="utf-8-sig")
print(f"✅ 예측 결과 저장 완료: {SAVE_PRED_RESULT}")
print("컬럼 예시:", ["pv_id", "time", "nins", "pred_nins", "abs_error"])

# =========================
# 11) 메모리 정리
# =========================
del X_train, X_test, y_train, y_test, raw_pred, pred
gc.collect()
