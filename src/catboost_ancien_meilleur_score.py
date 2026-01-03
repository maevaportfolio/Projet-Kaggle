"""
================================================================================
🦠 MODÈLE ANTI-OVERFIT - Feature Selection + Forte Régularisation
================================================================================
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("="*80)
print("🦠 MODÈLE ANTI-OVERFIT")
print("="*80)

# Chargement
train = pd.read_csv('train_enriched.csv')
test = pd.read_csv('test_enriched.csv')
train = train.sort_values('week').reset_index(drop=True)
test = test.sort_values('week').reset_index(drop=True)

# =============================================================================
# FEATURE ENGINEERING MODÉRÉ (moins de features, plus stables)
# =============================================================================

def create_features_stable(df, hist_data):
    """Features stables, moins d'overfit"""
    df = df.copy()
    hist = hist_data.copy()
    
    df['year'] = df['week'].astype(str).str[:4].astype(int)
    df['week_num'] = df['week'].astype(str).str[4:].astype(int)
    hist['year'] = hist['week'].astype(str).str[:4].astype(int)
    hist['week_num'] = hist['week'].astype(str).str[4:].astype(int)
    
    # Fourier (seulement 3 harmoniques)
    for k in [1, 2, 3]:
        df[f'sin_{k}'] = np.sin(2 * np.pi * k * df['week_num'] / 52)
        df[f'cos_{k}'] = np.cos(2 * np.pi * k * df['week_num'] / 52)
    
    # Phases simples
    df['is_peak'] = df['week_num'].isin([4, 5, 6, 7, 8]).astype(int)
    df['is_rise'] = df['week_num'].isin(list(range(48, 53)) + [1, 2, 3]).astype(int)
    df['is_low'] = df['week_num'].isin(range(18, 45)).astype(int)
    
    df['dist_peak'] = df['week_num'].apply(lambda w: min(abs(w - 5), abs(w - 57), abs(w + 47)))
    df['peak_intensity'] = np.exp(-df['dist_peak'] / 5)
    
    # Stats région × semaine (ESSENTIELLES)
    agg_rw = hist.groupby(['region_code', 'week_num'])['TauxGrippe'].agg([
        'mean', 'std', 'median',
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]).reset_index()
    agg_rw.columns = ['region_code', 'week_num', 'rw_mean', 'rw_std', 'rw_median', 'rw_q25', 'rw_q75']
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
    
    # Quelques dérivées stables
    df['rw_iqr'] = df['rw_q75'] - df['rw_q25']
    df['rw_vs_w'] = df['rw_mean'] / (df['w_mean'] + 1)
    
    # Google (TRÈS IMPORTANT)
    for col in ['requete_grippe', 'requete_grippe_aviaire_vaccin', 'requete_grippe_aviaire_vaccin_porcine_porc_H1N1_AH1N1_A_mexicaine_Mexique_pandemie']:
        df[col] = df[col].fillna(0)
    
    df['google_log'] = np.log1p(df['requete_grippe_aviaire_vaccin_porcine_porc_H1N1_AH1N1_A_mexicaine_Mexique_pandemie'])
    
    google_mean = hist['requete_grippe_aviaire_vaccin_porcine_porc_H1N1_AH1N1_A_mexicaine_Mexique_pandemie'].mean()
    df['google_anomaly'] = df['requete_grippe_aviaire_vaccin_porcine_porc_H1N1_AH1N1_A_mexicaine_Mexique_pandemie'] / (google_mean + 1)
    
    # Interactions clés (les plus stables)
    df['google_x_rw'] = df['google_log'] * df['rw_mean']
    df['google_x_w'] = df['google_log'] * df['w_mean']
    df['google_anom_x_rw'] = df['google_anomaly'] * df['rw_mean']
    
    # Météo simple
    if 't' in df.columns:
        df['t'] = df['t'].fillna(df['t'].median())
        df['cold'] = np.clip(8 - df['t'], 0, None)
    
    if 'rr1' in df.columns:
        df['rr1'] = df['rr1'].fillna(0)
    
    # Region encoding
    df['region_encoded'] = pd.factorize(df['region_code'])[0]
    
    return df

# =============================================================================
# PRÉPARATION
# =============================================================================
print("\n🔧 Features stables...")

split_idx = int(len(train) * 0.80)
train_data = train.iloc[:split_idx].copy()
val_data = train.iloc[split_idx:].copy()

train_feat = create_features_stable(train_data, train_data)
val_feat = create_features_stable(val_data, train_data)
test_feat = create_features_stable(test, train)
full_feat = create_features_stable(train, train)

exclude = ['Id', 'week', 'region_name', 'TauxGrippe', 'franche_comte_impute', 
           'year', 'annee', 'mois', 'region_code', 'requete_grippe', 'TauxGrippe_google']
features = [c for c in train_feat.columns if c not in exclude 
            and train_feat[c].dtype in ['float64', 'int64']]

print(f"   Features: {len(features)}")

X_train = train_feat[features + ['region_code']]
y_train = train_feat['TauxGrippe']
X_val = val_feat[features + ['region_code']]
y_val = val_feat['TauxGrippe']
X_full = full_feat[features + ['region_code']]
y_full = full_feat['TauxGrippe']
X_test = test_feat[features + ['region_code']]

# =============================================================================
# MODÈLE GLOBAL TRÈS RÉGULARISÉ
# =============================================================================
print("\n" + "="*80)
print("MODÈLE GLOBAL TRÈS RÉGULARISÉ")
print("="*80)

# Tester plusieurs configs très conservatrices
configs = [
    {'depth': 4, 'lr': 0.03, 'l2': 10, 'min_data': 50},
    {'depth': 5, 'lr': 0.02, 'l2': 10, 'min_data': 50},
    {'depth': 4, 'lr': 0.05, 'l2': 15, 'min_data': 60},
    {'depth': 5, 'lr': 0.03, 'l2': 8, 'min_data': 40},
    {'depth': 3, 'lr': 0.05, 'l2': 10, 'min_data': 50},
    {'depth': 4, 'lr': 0.02, 'l2': 20, 'min_data': 70},
]

best = {'rmse': float('inf'), 'config': None, 'model': None}

for cfg in configs:
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=cfg['lr'],
        depth=cfg['depth'],
        l2_leaf_reg=cfg['l2'],
        min_data_in_leaf=cfg['min_data'],
        random_strength=1.5,
        bagging_temperature=1.0,
        border_count=64,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        use_best_model=True
    )
    
    train_pool = Pool(X_train, y_train, cat_features=['region_code'])
    val_pool = Pool(X_val, y_val, cat_features=['region_code'])
    model.fit(train_pool, eval_set=val_pool)
    
    pred = np.clip(model.predict(X_val), 0, None)
    score = rmse(y_val, pred)
    
    # Aussi calculer le train RMSE pour voir l'overfit
    pred_train = np.clip(model.predict(X_train), 0, None)
    train_score = rmse(y_train, pred_train)
    gap = train_score - score
    
    print(f"   d={cfg['depth']}, lr={cfg['lr']}, l2={cfg['l2']}, min={cfg['min_data']} → Val={score:.2f}, Train={train_score:.2f}, Gap={gap:+.1f}")
    
    if score < best['rmse']:
        best = {'rmse': score, 'config': cfg, 'model': model, 'iter': model.get_best_iteration()}

print(f"\n🏆 Best: Val RMSE={best['rmse']:.2f}")

# =============================================================================
# MODÈLE PAR RÉGION TRÈS RÉGULARISÉ
# =============================================================================
print("\n" + "="*80)
print("MODÈLES RÉGIONAUX TRÈS RÉGULARISÉS")
print("="*80)

regions = train['region_code'].unique()
regional_preds_val = np.zeros(len(val_feat))
regional_models = {}

for region in sorted(regions):
    train_mask = train_feat['region_code'] == region
    val_mask = val_feat['region_code'] == region
    
    X_train_r = train_feat.loc[train_mask, features].values
    y_train_r = train_feat.loc[train_mask, 'TauxGrippe'].values
    X_val_r = val_feat.loc[val_mask, features].values
    y_val_r = val_feat.loc[val_mask, 'TauxGrippe'].values
    
    # Modèle TRÈS régularisé pour éviter overfit
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=4,  # Peu profond
        l2_leaf_reg=15,  # Forte régularisation
        min_data_in_leaf=20,
        random_strength=2.0,
        bagging_temperature=1.0,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    if len(X_val_r) > 0:
        model.fit(X_train_r, y_train_r, eval_set=(X_val_r, y_val_r))
        pred_val_r = np.clip(model.predict(X_val_r), 0, None)
        regional_preds_val[val_mask] = pred_val_r
        region_rmse = rmse(y_val_r, pred_val_r)
    else:
        model.fit(X_train_r, y_train_r)
        region_rmse = 0
    
    regional_models[region] = model
    print(f"   Région {region}: RMSE={region_rmse:.2f}")

regional_rmse = rmse(y_val, regional_preds_val)
print(f"\n📊 Régional global: {regional_rmse:.2f}")

# =============================================================================
# BLENDING OPTIMAL
# =============================================================================
print("\n" + "="*80)
print("BLENDING")
print("="*80)

global_pred_val = np.clip(best['model'].predict(X_val), 0, None)

best_w, best_blend = 0.5, float('inf')
for w in np.arange(0.0, 1.05, 0.05):
    blend = w * regional_preds_val + (1 - w) * global_pred_val
    score = rmse(y_val, blend)
    if score < best_blend:
        best_blend = score
        best_w = w

print(f"\n📊 Global seul:    {best['rmse']:.2f}")
print(f"📊 Régional seul:  {regional_rmse:.2f}")
print(f"📊 Blend optimal:  {best_blend:.2f} (w_reg={best_w:.2f})")

# =============================================================================
# PRÉDICTIONS TEST
# =============================================================================
print("\n" + "="*80)
print("PRÉDICTIONS TEST")
print("="*80)

# Global final
cfg = best['config']
global_final = CatBoostRegressor(
    iterations=best['iter'] + 50,
    learning_rate=cfg['lr'],
    depth=cfg['depth'],
    l2_leaf_reg=cfg['l2'],
    min_data_in_leaf=cfg['min_data'],
    random_strength=1.5,
    bagging_temperature=1.0,
    border_count=64,
    random_seed=42,
    verbose=0
)
full_pool = Pool(X_full, y_full, cat_features=['region_code'])
global_final.fit(full_pool)
global_pred_test = np.clip(global_final.predict(X_test), 0, None)

# Régional final
regional_pred_test = np.zeros(len(test_feat))
for region in sorted(regions):
    full_mask = full_feat['region_code'] == region
    test_mask = test_feat['region_code'] == region
    
    X_full_r = full_feat.loc[full_mask, features].values
    y_full_r = full_feat.loc[full_mask, 'TauxGrippe'].values
    X_test_r = test_feat.loc[test_mask, features].values
    
    model = CatBoostRegressor(
        iterations=regional_models[region].get_best_iteration() + 30,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=15,
        min_data_in_leaf=20,
        random_strength=2.0,
        bagging_temperature=1.0,
        random_seed=42,
        verbose=0
    )
    model.fit(X_full_r, y_full_r)
    
    if len(X_test_r) > 0:
        regional_pred_test[test_mask] = np.clip(model.predict(X_test_r), 0, None)

# Blend
blend_pred_test = best_w * regional_pred_test + (1 - best_w) * global_pred_test

print(f"\n📊 Stats prédictions:")
print(f"   Global - Mean: {global_pred_test.mean():.2f}")
print(f"   Régional - Mean: {regional_pred_test.mean():.2f}")
print(f"   Blend - Mean: {blend_pred_test.mean():.2f}")

# =============================================================================
# SUBMISSIONS
# =============================================================================
# Global seul (le plus sûr anti-overfit)
sub_global = pd.DataFrame({
    'Id': test['Id'].astype(int),
    'TauxGrippe': global_pred_test
}).sort_values('Id').reset_index(drop=True)

path1 = 'submission_global_conservative.csv'
sub_global.to_csv(path1, index=False)

# Blend
sub_blend = pd.DataFrame({
    'Id': test['Id'].astype(int),
    'TauxGrippe': blend_pred_test
}).sort_values('Id').reset_index(drop=True)

path2 = 'submission_blend_conservative.csv'
sub_blend.to_csv(path2, index=False)

# Régional seul
sub_reg = pd.DataFrame({
    'Id': test['Id'].astype(int),
    'TauxGrippe': regional_pred_test
}).sort_values('Id').reset_index(drop=True)

path3 = 'submission_regional_conservative.csv'
sub_reg.to_csv(path3, index=False)

print(f"\n✅ {path1}")
print(f"✅ {path2}")
print(f"✅ {path3}")

print(f"""
================================================================================
✅ RÉSUMÉ
================================================================================
📊 Features: {len(features)} (conservateur)

📊 Validation:
   Global:   {best['rmse']:.2f}
   Régional: {regional_rmse:.2f}
   Blend:    {best_blend:.2f} (w_reg={best_w:.2f})

💡 Soumets submission_global_conservative.csv en premier (le plus stable)
================================================================================
""")