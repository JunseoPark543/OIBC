import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import time as t
import traceback
'''
보간, 낮밤 학습, 밤 = 0 처리 x
'''
# =========================
# 1) Load
# =========================
start = t.time()
data = pd.read_csv("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA/train.csv")
end = t.time()
print(f"[IO] 데이터 불러오기 완료: {end-start:.2f}s, shape={data.shape}")

# =========================
# 2) Datetime 변환
# =========================
start = t.time()
data['time'] = pd.to_datetime(data['time'], errors='coerce')
end = t.time()
print(f"[Time] time 변환 완료: {end-start:.2f}s, 결측 {data['time'].isna().sum()}개")

# =========================
# 3) 보간 규칙 정의
# =========================
cubic_cols = [
    "temp_a", "temp_b", "appr_temp", "real_feel_temp", "real_feel_temp_shade",
    "wind_chill_temp", "temp_max", "temp_min", "dew_point", "rel_hum",
    "humidity", "pressure", "ground_press", "vis", "uv_idx"
]
step_ffill_cols = ["precip_1h", "rain", "snow"]

exclude_from_interp = {"time", "pv_id", "type", "energy", "nins"}

def _coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def impute_by_rules(group_df: pd.DataFrame, pv_id: str) -> pd.DataFrame:
    """pv_id 그룹별로 시간순 정렬 후 보간 적용"""
    print(f"\n▶ [Group] 시작: {pv_id} ({len(group_df)}행)")
    g = group_df.sort_values("time").copy()

    present_cubic = [c for c in cubic_cols if c in g.columns]
    present_step  = [c for c in step_ffill_cols if c in g.columns]
    g = _coerce_numeric(g, present_cubic + present_step)

    # 3차 스플라인
    if present_cubic:
        try:
            print(f"  - cubic_spline 적용: {present_cubic[:5]}{'...' if len(present_cubic)>5 else ''}")
            g[present_cubic] = g[present_cubic].interpolate(
                method="spline", order=3, limit_direction="both"
            )
        except Exception as e:
            print(f"  [WARN] spline 실패 → polynomial fallback ({pv_id}): {e}")
            traceback.print_exc(limit=1)
            g[present_cubic] = g[present_cubic].interpolate(
                method="polynomial", order=3, limit_direction="both"
            )

    # 계단형 (앞방향 보간)
    if present_step:
        print(f"  - step_ffill 적용: {present_step}")
        g[present_step] = g[present_step].ffill()

    # 나머지 수치 피처: 선형 보간
    numeric_cols = g.select_dtypes(include=[np.number]).columns.tolist()
    other_linear_cols = [
        c for c in numeric_cols
        if (c not in present_cubic)
        and (c not in present_step)
        and (c not in exclude_from_interp)
    ]
    if other_linear_cols:
        print(f"  - linear 적용 ({len(other_linear_cols)}개): {other_linear_cols[:5]}{'...' if len(other_linear_cols)>5 else ''}")
        try:
            g[other_linear_cols] = g[other_linear_cols].interpolate(
                method="linear", limit_direction="both"
            )
        except Exception as e:
            print(f"  [ERROR] linear 보간 중 오류 ({pv_id}): {e}")
            traceback.print_exc(limit=1)

    print(f"  ✓ 완료: {pv_id}")
    return g

# =========================
# 4) 전체 데이터에 보간 적용
# =========================
start = t.time()
groups = data["pv_id"].unique().tolist()
print(f"[Impute] 총 {len(groups)}개 pv_id 그룹 처리 시작")

imputed_dfs = []
for i, pv in enumerate(groups, 1):
    try:
        imputed = impute_by_rules(data[data["pv_id"] == pv], pv)
        imputed_dfs.append(imputed)
    except Exception as e:
        print(f"[ERROR] {pv} 그룹 보간 실패: {e}")
        traceback.print_exc(limit=1)
    if i % 5 == 0:
        print(f"  -> 진행률: {i}/{len(groups)} ({i/len(groups)*100:.1f}%)")

data_imputed = pd.concat(imputed_dfs, ignore_index=True)
end = t.time()
print(f"[Impute] 전체 보간 완료: {end-start:.2f}s, shape={data_imputed.shape}")

# =========================
# 5) Train/Test 분리
# =========================
test_pv_ids_int = [177,178,179,181,184,185,186,187,188,189,190,191,193,195,196,197,198,199,200,201,202,203,205,206,207,208,209]
test_pv_ids = [f"PV_ID_{i}" for i in test_pv_ids_int]

test_df  = data_imputed[data_imputed["pv_id"].isin(test_pv_ids)].copy()
train_df = data_imputed[~data_imputed["pv_id"].isin(test_pv_ids)].copy()
print(f"[Split] train={train_df.shape}, test={test_df.shape}")

# =========================
# 6) Feature 선택
# =========================
features = data_imputed.columns.drop(['time', 'pv_id', 'type', 'energy', 'nins'])
print(f"[Feat] 총 {len(features)}개 피처 사용")

# 결측이 남은 경우 제거
null_counts = train_df[features].isna().sum().sum()
print(f"[Check] train 결측치 총합: {null_counts}")
train_df = train_df.dropna(subset=features)

# =========================
# 7) 모델 학습/평가
# =========================
print("[Train] LGBMRegressor 학습 시작")
lgbm = LGBMRegressor(random_state=42)
lgbm.fit(train_df[features], train_df['nins'])

lgbm_pred = lgbm.predict(test_df[features])
lgbm_pred = np.where(lgbm_pred < 0, 0, lgbm_pred)

y_true = test_df['nins']
mae = mean_absolute_error(y_true, lgbm_pred)
print(f"[Eval] MAE: {mae:.6f}")
