import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time as t

# CSV 파일 불러오기 (파일 경로 수정 필요)
start = t.time()
data = pd.read_csv("C:/Users/pc/Desktop/OIBC/OIBC_2025_DATA/train.csv")
end = t.time()
print(f'데이터 불러오기 걸린 시간:{end-start}')

# datetime
start = t.time()
data['time'] = pd.to_datetime(data['time'])
end = t.time()
print(f'time 변환 걸린 시간:{end-start}')

test_pv_ids_int = [177,178,179,181,184,185,186,187,188,189,190,191,193,195,196,197,198,199,200,201,202,203,205,206,207,208,209]
test_pv_ids = [f"PV_ID_{i}" for i in test_pv_ids_int]  # <- 접두사 맞추기!

data["pv_id"] = data["pv_id"].astype(str)  # 이미 object지만 확실히
test_df  = data[data["pv_id"].isin(test_pv_ids)].copy()
train_df = data[~data["pv_id"].isin(test_pv_ids)].copy()
print("train rows:", len(train_df), "test rows:", len(test_df))


print("pv_id dtype:", data['pv_id'].dtype)
print("unique pv ids in data (head):", data['pv_id'].astype(str).unique()[:10])
print("train_df rows:", len(train_df), "test_df rows:", len(test_df))

features = data.columns.drop(['time', 'pv_id', 'type', 'energy', 'nins'])
print(features)

dfs = []
for pv_id, df_group in train_df.groupby('pv_id'):
    dfs.append(df_group[features].apply(lambda x: x.bfill()))
# concat: 데이터 합치기 
print("dfs length before concat (train):", len(dfs))
weather_fillna_df = pd.concat(dfs)
train_df[features] = weather_fillna_df

dfs = []
for pv_id, df_group in test_df.groupby('pv_id'):
    dfs.append(df_group[features].apply(lambda x: x.bfill()))
print("dfs length before concat (train):", len(dfs))
weather_fillna_df = pd.concat(dfs)
test_df[features] = weather_fillna_df

train_df = train_df.dropna(subset=features)

lgbm = LGBMRegressor()
lgbm.fit(train_df[features], train_df['nins'])
lgbm_pred = lgbm.predict(test_df[features])
lgbm_pred[lgbm_pred < 0] = 0

y_true = test_df['nins']

mae = mean_absolute_error(y_true, lgbm_pred)
print(f"MAE: {mae}")
