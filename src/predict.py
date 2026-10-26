import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. DATA LOADING & AGGREGATION
# ==========================================
def load_and_aggregate_data(data_dir: str = "challenge_data") -> pd.DataFrame:
    print("📂 Loading and aggregating data...")
    data_path = Path(data_dir)
    df_readings = pd.read_csv(data_path / "vm_cpu_readings.csv.gz")
    
    # Target 1: active_vm_count
    active_vms = df_readings.groupby("timestamp")["vm_id"].nunique().reset_index()
    active_vms.rename(columns={"vm_id": "active_vm_count"}, inplace=True)
    
    # Target 2: aggregate_cpu_utilization
    agg_cpu = df_readings.groupby("timestamp")["avg_cpu"].mean().reset_index()
    agg_cpu.rename(columns={"avg_cpu": "aggregate_cpu_utilization"}, inplace=True)
    
    df = pd.merge(active_vms, agg_cpu, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def create_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Creates lag and rolling features for a specific target."""
    df = df.copy()
    # Lags (past values)
    for lag in [1, 2, 3, 4, 5, 6, 12, 24]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    df[f'{target_col}_roll_mean_6'] = df[target_col].shift(1).rolling(window=6).mean()
    df[f'{target_col}_roll_std_6'] = df[target_col].shift(1).rolling(window=6).std()
    
    return df

# ==========================================
# 3. MODELING & PREDICTION
# ==========================================
def recursive_predict(train_df, feature_cols, target_col, steps=24):
    """Predicts multiple steps ahead using LightGBM recursively."""
    # Train LightGBM model
    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, target_col]
    
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Recursive prediction
    predictions = []
    # We need a working copy of the last few rows to calculate lags for the next step
    history = train_df[[target_col] + feature_cols].copy()
    
    for step in range(steps):
        # Get the last row's features
        last_row_features = history[feature_cols].iloc[[-1]]
        
        # Predict next value
        next_val = model.predict(last_row_features)[0]
        
        # Ensure constraints (active_vm_count >= 0, cpu in [0,100])
        if target_col == 'active_vm_count':
            next_val = max(0, round(next_val))
        elif target_col == 'aggregate_cpu_utilization':
            next_val = max(0.0, min(100.0, next_val))
            
        predictions.append(next_val)
        
        # Update history for the next iteration (shift features)
        new_row = history.iloc[[-1]].copy()
        new_row[target_col] = next_val
        
        # Update lag features manually for the new row
        for lag in [1, 2, 3, 4, 5, 6, 12, 24]:
            if lag == 1:
                new_row[f'{target_col}_lag_{lag}'] = next_val
            else:
                # Look back in the history
                lookback_idx = -(lag)
                if abs(lookback_idx) <= len(history):
                    new_row[f'{target_col}_lag_{lag}'] = history[target_col].iloc[lookback_idx]
                else:
                    new_row[f'{target_col}_lag_{lag}'] = np.nan
                    
        # Update rolling features
        recent_vals = list(history[target_col].iloc[-5:]) + [next_val]
        new_row[f'{target_col}_roll_mean_6'] = np.mean(recent_vals)
        new_row[f'{target_col}_roll_std_6'] = np.std(recent_vals)
        
        history = pd.concat([history, new_row], ignore_index=True)
        
    return predictions

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Load data
    df = load_and_aggregate_data("challenge_data")
    
    # Split data (110 train, 24 holdout)
    train_df = df.iloc[:110].copy()
    holdout_df = df.iloc[110:].copy()
    holdout_timestamps = holdout_df['timestamp'].values
    
    targets = ['active_vm_count', 'aggregate_cpu_utilization']
    final_predictions = {'timestamp': holdout_timestamps}
    
    for target in targets:
        print(f"\n🚀 Processing target: {target}")
        
        # Create features
        df_feat = create_features(df, target)
        train_feat = df_feat.iloc[:110]
        
        # Define feature columns (exclude timestamp and target)
        feature_cols = [col for col in df_feat.columns if col not in ['timestamp', target]]
        
        # --- BASELINE: Last Value (Naive) ---
        baseline_pred = [train_df[target].iloc[-1]] * 24
        print(f" Baseline (Naive) first 3 preds: {baseline_pred[:3]}")
        
        # --- ML MODEL: LightGBM Recursive ---
        ml_preds = recursive_predict(train_feat, feature_cols, target, steps=24)
        print(f"🤖 LightGBM first 3 preds: {[round(p, 2) for p in ml_preds[:3]]}")
        
        # We will use the LightGBM predictions for the final submission
        final_predictions[target] = ml_preds

    # ==========================================
    # 5. SAVE PREDICTIONS
    # ==========================================
    pred_df = pd.DataFrame(final_predictions)
    pred_df.to_csv("predictions.csv", index=False)
    print("\n✅ Successfully saved predictions.csv!")
    print(pred_df)