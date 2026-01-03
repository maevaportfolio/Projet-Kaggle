"""
================================================================================
🦠 CATBOOST FINAL - MEILLEURE VERSION
================================================================================
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("="*80)
print("🦠 CATBOOST FINAL")
print("="*80)

# =============================================================================
# CHARGEMENT
# =============================================================================
train = pd.read_csv('/mnt/user-data/uploads/train_complet_avec_imputation.csv')
test = pd.read_csv('/mnt/user-data/uploads/test_complet_avec_imputation.csv')

train = train.sort_values('week').reset_index(drop=True)
test = test.sort_values('week').reset_index(drop=True)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df, hist_data):
    df = df.copy()
    hist = hist_data.copy()
    
    df['year'] = df['week'].astype(str).str[:4].astype(int)
    df['week_num'] = df['week'].astype(str).str[4:].astype(int)
    hist['year'] = hist['week'].astype(str).str[:4].astype(int)
    hist['week_num'] = hist['week'].astype(str).str[4:].astype(int)
    
    # Fourier
    for k in [1, 2, 3]:
        df[f'sin_{k}'] = np.sin(2 * np.pi * k * df['week_num'] / 52)
        df[f'cos_{k}'] = np.cos(2 * np.pi * k * df['week_num'] / 52)
    
    # Phases
    df['is_peak'] = df['week_num'].isin([4, 5, 6, 7, 8]).astype(int)
    df['is_rise'] = df['week_num'].isin(list(range(48, 53)) + [1, 2, 3]).astype(int)
    df['is_low'] = df['week_num'].isin(range(18, 45)).astype(int)
    
    df['dist_peak'] = df['week_num'].apply(lambda w: min(abs(w - 5), abs(w - 57), abs(w + 47)))
    df['peak_intensity'] = np.exp(-df['dist_peak'] / 5)
    
    # Stats région × semaine
    agg_rw = hist.groupby(['region_code', 'week_num'])['TauxGrippe'].agg([
        'mean', 'std', 'median', 'min', 'max',
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
        ('q90', lambda x: x.quantile(0.9))
    ]).reset_index()
    agg_rw.columns = ['region_code', 'week_num'] + [f'rw_{c}' for c in ['mean', 'std', 'median', 'min', 'max', 'q25', 'q75', 'q90']]
    df = df.merge(agg_rw, on=['region_code', 'week_num'], how='left')
    
    # Stats nationales
    agg_w = hist.groupby('week_num')['TauxGrippe'].agg(['mean', 'std', 'median']).reset_index()
    agg_w.columns = ['week_num', 'w_mean', 'w_std', 'w_median']
    df = df.merge(agg_w, on='week_num', how='left')
    
    # Stats région
    agg_r = hist.groupby('region_code')['TauxGrippe'].agg(['mean', 'std']).reset_index()
    agg_r.columns = ['region_code', 'r_mean', 'r_std']
    df = df.merge(agg_r, on='region_code', how='left')
    
    # Imputation
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    # Features dérivées
    df['rw_iqr'] = df['rw_q75'] - df['rw_q25']
    df['rw_range'] = df['rw_max'] - df['rw_min']
    df['rw_vs_w'] = df['rw_mean'] / (df['w_mean'] + 1)
    
    # Google
    for col in ['google_grippe', 'google_grippe_filtre1', 'google_grippe_filtre2']:
        df[col] = df[col].fillna(0)
    
    df['google_log'] = np.log1p(df['google_grippe_filtre2'])
    df['google_sqrt'] = np.sqrt(df['google_grippe_filtre2'])
    
    google_mean = hist['google_grippe_filtre2'].mean()
    df['google_anomaly'] = df['google_grippe_filtre2'] / (google_mean + 1)
    
    # Interactions Google
    df['google_x_rw'] = df['google_log'] * df['rw_mean']
    df['google_x_w'] = df['google_log'] * df['w_mean']
    df['google_x_peak'] = df['google_log'] * df['peak_intensity']
    df['google_anom_x_rw'] = df['google_anomaly'] * df['rw_mean']
    df['google_x_is_peak'] = df['google_grippe_filtre2'] * df['is_peak']
    
    # Météo
    for col in ['t', 'td', 'u', 'ff', 'vv', 'rr1']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    if 't' in df.columns:
        df['cold'] = np.clip(8 - df['t'], 0, None)
        df['is_cold'] = (df['t'] < 5).astype(int)
        df['cold_x_peak'] = df['cold'] * df['peak_intensity']
        df['t_x_rw'] = df['t'] * df['rw_mean']
    
    # Démo
    if 'ensemble_total' in df.columns:
        for col in ['ensemble_total', 'ensemble_60-74', 'ensemble_75+']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        df['pop_log'] = np.log1p(df['ensemble_total'])
        df['pct_elderly'] = (df['ensemble_60-74'] + df['ensemble_75+']) / (df['ensemble_total'] + 1)
    
    # Interactions historiques
    df['rw_x_peak'] = df['rw_mean'] * df['is_peak']
    df['rw_x_rise'] = df['rw_mean'] * df['is_rise']
    
    return df

# =============================================================================
# PRÉPARATION
# =============================================================================
print("\n🔧 Préparation...")

split_idx = int(len(train) * 0.8)
train_data = train.iloc[:split_idx].copy()
val_data = train.iloc[split_idx:].copy()

train_feat = create_features(train_data, train_data)
val_feat = create_features(val_data, train_data)
test_feat = create_features(test, train)
full_feat = create_features(train, train)

exclude = ['Id', 'week', 'region_name', 'TauxGrippe', 'franche_comte_impute', 
           'year', 'annee', 'mois', 'region_code']
features = [c for c in train_feat.columns if c not in exclude 
            and train_feat[c].dtype in ['float64', 'int64']]
all_features = features + ['region_code']
cat_features = ['region_code']

print(f"   Features: {len(all_features)}")

X_train = train_feat[all_features]
y_train = train_feat['TauxGrippe']
X_val = val_feat[all_features]
y_val = val_feat['TauxGrippe']
X_test = test_feat[all_features]
X_full = full_feat[all_features]
y_full = full_feat['TauxGrippe']

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# =============================================================================
# CATBOOST AVEC HYPERPARAMÈTRES OPTIMISÉS
# =============================================================================
print("\n" + "="*80)
print("CATBOOST OPTIMISÉ")
print("="*80)

# Configuration qui a donné 98.85
model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.01,
    depth=8,
    l2_leaf_reg=1,
    min_data_in_leaf=20,
    random_strength=1.0,
    bagging_temperature=0.5,
    border_count=128,
    grow_policy='SymmetricTree',
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=150,
    use_best_model=True
)

print("\n🚀 Entraînement...")
model.fit(train_pool, eval_set=val_pool)

pred_val = np.clip(model.predict(X_val), 0, None)
pred_train = np.clip(model.predict(X_train), 0, None)

val_rmse = rmse(y_val, pred_val)
train_rmse = rmse(y_train, pred_train)
val_mae = mean_absolute_error(y_val, pred_val)
val_r2 = r2_score(y_val, pred_val)

print(f"\n{'='*60}")
print(f"📊 RÉSULTATS VALIDATION")
print(f"{'='*60}")
print(f"   Train RMSE: {train_rmse:.2f}")
print(f"   Val RMSE:   {val_rmse:.2f}")
print(f"   Val MAE:    {val_mae:.2f}")
print(f"   Val R²:     {val_r2:.4f}")
print(f"   Best iter:  {model.get_best_iteration()}")

# =============================================================================
# ENTRAÎNEMENT FINAL
# =============================================================================
print("\n" + "="*80)
print("ENTRAÎNEMENT FINAL SUR TOUT LE TRAIN")
print("="*80)

best_iter = model.get_best_iteration()

model_final = CatBoostRegressor(
    iterations=best_iter + 100,
    learning_rate=0.01,
    depth=8,
    l2_leaf_reg=1,
    min_data_in_leaf=20,
    random_strength=1.0,
    bagging_temperature=0.5,
    border_count=128,
    grow_policy='SymmetricTree',
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)

full_pool = Pool(X_full, y_full, cat_features=cat_features)
model_final.fit(full_pool)

test_pred = np.clip(model_final.predict(X_test), 0, None)

print(f"\n📊 Statistiques prédictions:")
print(f"   Min:    {test_pred.min():.2f}")
print(f"   Median: {np.median(test_pred):.2f}")
print(f"   Mean:   {test_pred.mean():.2f}")
print(f"   Max:    {test_pred.max():.2f}")

# =============================================================================
# SUBMISSION
# =============================================================================
submission = pd.DataFrame({
    'Id': test['Id'].astype(int),
    'TauxGrippe': test_pred
}).sort_values('Id').reset_index(drop=True)

path = '/mnt/user-data/outputs/submission_catboost_final.csv'
submission.to_csv(path, index=False)
print(f"\n✅ {path}")

print("\n📋 Aperçu:")
print(submission.head(10))

# Feature importance
print("\n" + "="*80)
print("TOP 20 FEATURES")
print("="*80)
imp = model.get_feature_importance()
imp_df = pd.DataFrame({'feature': all_features, 'importance': imp})
imp_df = imp_df.sort_values('importance', ascending=False)
for _, row in imp_df.head(20).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.2f}")

print(f"""
================================================================================
✅ RÉSUMÉ FINAL
================================================================================
🏆 Val RMSE: {val_rmse:.2f}
🏆 Val MAE:  {val_mae:.2f}
🏆 Val R²:   {val_r2:.4f}

💡 RMSE Kaggle attendu: ~{val_rmse:.0f}-{val_rmse*1.05:.0f}
================================================================================
""")