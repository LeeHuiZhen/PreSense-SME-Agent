import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────────────────────────────────────
RUL_CAP  = 125
DATA_DIR = 'archive/'
COLS = (
    ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']
    + [f'sensor_{i}' for i in range(1, 22)]
)

# These 7 sensors are near-constant in FD001 — they add noise, not signal
DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                'sensor_16', 'sensor_18', 'sensor_19']
FEATURES = [f'sensor_{i}' for i in range(1, 22)
            if f'sensor_{i}' not in DROP_SENSORS]

print("=" * 60)
print("  PreSense — Model Retraining Script")
print("=" * 60)

# ── LOAD DATA ───────────────────────────────────────────────────────────────
print("\n[1/6] Loading data from archive/...")
df_train = pd.read_csv(f'{DATA_DIR}train_FD001.txt',
                       sep=r'\s+', header=None, names=COLS)
df_test  = pd.read_csv(f'{DATA_DIR}test_FD001.txt',
                       sep=r'\s+', header=None, names=COLS)
rul_true = pd.read_csv(f'{DATA_DIR}RUL_FD001.txt',
                       header=None, names=['RUL'])
print(f"  Train: {len(df_train):,} rows, {df_train['unit_number'].nunique()} engines")
print(f"  Test : {len(df_test):,} rows,  {df_test['unit_number'].nunique()} engines")

# ── PIECEWISE CAPPED RUL LABELS ─────────────────────────────────────────────
# Key improvement: cap at 125 — engines are "healthy" above this,
# so we don't penalise the model for not predicting far-future cycles.
print("\n[2/6] Computing piecewise capped RUL labels (cap=125)...")
def add_rul(df, cap=RUL_CAP):
    rul = (df.groupby('unit_number')['time_in_cycles']
             .transform('max') - df['time_in_cycles'])
    df['RUL'] = rul.clip(upper=cap)
    return df

df_train = add_rul(df_train)
print(f"  RUL range: {df_train['RUL'].min():.0f} – {df_train['RUL'].max():.0f} cycles")

# ── FEATURE ENGINEERING ─────────────────────────────────────────────────────
# Rolling mean (window=3) smooths sensor noise without leaking future data
print("\n[3/6] Engineering features (rolling mean window=3)...")
for col in FEATURES:
    df_train[f'{col}_roll3'] = (
        df_train.groupby('unit_number')[col]
                .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

FINAL_FEATURES = FEATURES + [f'{f}_roll3' for f in FEATURES]
print(f"  Total features: {len(FINAL_FEATURES)}")

# ── TRAIN / VAL SPLIT ───────────────────────────────────────────────────────
print("\n[4/6] Splitting train/validation (80/20)...")
X = df_train[FINAL_FEATURES]
y = df_train['RUL']
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler    = StandardScaler()
X_tr_s    = scaler.fit_transform(X_tr)
X_val_s   = scaler.transform(X_val)

# ── TRAIN MODELS ────────────────────────────────────────────────────────────
print("\n[5/6] Training models...")

# Random Forest
print("  Training Random Forest (200 trees)...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_tr_s, y_tr)
val_rmse_rf = np.sqrt(mean_squared_error(y_val, rf.predict(X_val_s)))
print(f"  → Random Forest   Val RMSE: {val_rmse_rf:.2f}")

# Gradient Boosting
print("  Training Gradient Boosting (300 estimators)...")
gb = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb.fit(X_tr_s, y_tr)
val_rmse_gb = np.sqrt(mean_squared_error(y_val, gb.predict(X_val_s)))
print(f"  → Gradient Boost  Val RMSE: {val_rmse_gb:.2f}")

# Pick winner
if val_rmse_rf <= val_rmse_gb:
    best_model = rf
    best_name  = "Random Forest"
    best_val   = val_rmse_rf
else:
    best_model = gb
    best_name  = "Gradient Boosting"
    best_val   = val_rmse_gb

print(f"\n  ✓ Selected: {best_name} (Val RMSE: {best_val:.2f})")

# ── OFFICIAL TEST SET EVALUATION ────────────────────────────────────────────
print("\n[6/6] Evaluating on official C-MAPSS FD001 test benchmark...")

# Apply rolling features to the full test set FIRST
for col in FEATURES:
    df_test[f'{col}_roll3'] = (
        df_test.groupby('unit_number')[col]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

# THEN extract the last row for evaluation
last_rows = df_test.groupby('unit_number').last().reset_index()

X_test_s = scaler.transform(last_rows[FINAL_FEATURES])
y_pred   = best_model.predict(X_test_s).clip(0, RUL_CAP)
y_true   = rul_true['RUL'].values

test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
improvement = ((41.66 - test_rmse) / 41.66) * 100

def cmapss_score(y_true, y_pred):
    d = y_pred - y_true
    return sum(np.exp(-e/13)-1 if e < 0 else np.exp(e/10)-1 for e in d)

test_score = cmapss_score(y_true, y_pred)

print(f"\n{'=' * 60}")
print(f"  OFFICIAL TEST RMSE : {test_rmse:.2f}  (was 41.66)")
print(f"  IMPROVEMENT        : {improvement:+.1f}%")
print(f"  OFFICIAL S-SCORE   : {test_score:.2f}  (lower is better)")
print(f"  MODEL              : {best_name}")
print(f"{'=' * 60}")

# ── SAVE ARTIFACTS ──────────────────────────────────────────────────────────
joblib.dump(best_model,     'rul_model.pkl')
joblib.dump(scaler,         'rul_scaler.pkl')
joblib.dump(FINAL_FEATURES, 'feature_names.pkl')

print(f"""
Saved:
  rul_model.pkl      ← new model ({best_name})
  rul_scaler.pkl     ← StandardScaler (required by app.py)
  feature_names.pkl  ← feature list (required by app.py)

Pitch this to judges:
  "Our model achieves RMSE {test_rmse:.2f} on the official
   NASA C-MAPSS FD001 benchmark test set."
""")