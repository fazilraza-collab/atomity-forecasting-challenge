import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

def get_features_and_target(df, target_col):
    df = df.copy()
    for lag in [1, 2, 3, 4, 5, 6, 12, 24]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    df[f'{target_col}_roll_mean_6'] = df[target_col].shift(1).rolling(window=6).mean()
    df[f'{target_col}_roll_std_6'] = df[target_col].shift(1).rolling(window=6).std()
    return df

def expanding_window_cv(train_df, target_col, initial_window=50, step=10, horizon=24):
    mae_ml, rmse_ml = [], []
    mae_base, rmse_base = [], []
    
    end_idx = len(train_df) - horizon 
    
    for start_idx in range(initial_window, end_idx, step):
        train_window = train_df.iloc[:start_idx]
        test_window = train_df.iloc[start_idx:start_idx + horizon]
        
        full_df = pd.concat([train_window, test_window])
        feat_df = get_features_and_target(full_df, target_col)
        
        train_feat = feat_df.iloc[:start_idx].dropna()
        test_feat = feat_df.iloc[start_idx:start_idx + horizon]
        
        feature_cols = [c for c in train_feat.columns if c not in ['timestamp', target_col]]
        
        X_train, y_train = train_feat[feature_cols], train_feat[target_col]
        X_test, y_test = test_feat[feature_cols], test_feat[target_col]
        
        # 1. BASELINE: Naive (Last Observation Carried Forward)
        last_train_val = train_feat[target_col].iloc[-1]
        base_preds = np.full(len(y_test), last_train_val)
        mae_base.append(mean_absolute_error(y_test, base_preds))
        rmse_base.append(np.sqrt(mean_squared_error(y_test, base_preds)))
        
        # 2. ML MODEL: LightGBM
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        ml_preds = model.predict(X_test)
        
        mae_ml.append(mean_absolute_error(y_test, ml_preds))
        rmse_ml.append(np.sqrt(mean_squared_error(y_test, ml_preds)))
        
    return (np.mean(mae_base), np.mean(rmse_base)), (np.mean(mae_ml), np.mean(rmse_ml))

if __name__ == "__main__":
    data_path = Path("challenge_data")
    df_readings = pd.read_csv(data_path / "vm_cpu_readings.csv.gz")
    
    active_vms = df_readings.groupby("timestamp")["vm_id"].nunique().reset_index()
    active_vms = active_vms.rename(columns={"vm_id": "active_vm_count"})
    agg_cpu = df_readings.groupby("timestamp")["avg_cpu"].mean().reset_index()
    agg_cpu = agg_cpu.rename(columns={"avg_cpu": "aggregate_cpu_utilization"})
    df = pd.merge(active_vms, agg_cpu, on="timestamp").sort_values("timestamp")
    
    train_df = df.iloc[:110]
    
    print("--- Expanding Window Cross-Validation (24-step ahead) ---")
    
    for target in ['active_vm_count', 'aggregate_cpu_utilization']:
        (b_mae, b_rmse), (m_mae, m_rmse) = expanding_window_cv(train_df, target)
        improvement_mae = ((b_mae - m_mae) / b_mae) * 100 if b_mae > 0 else 0
        
        print(f"\nTarget: {target}")
        print(f"  Naive Baseline  -> MAE: {b_mae:.4f} | RMSE: {b_rmse:.4f}")
        print(f"  LightGBM Model  -> MAE: {m_mae:.4f} | RMSE: {m_rmse:.4f}")
        print(f"  Improvement     -> MAE reduced by {improvement_mae:.1f}%")