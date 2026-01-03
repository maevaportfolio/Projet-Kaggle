
import pandas as pd
import numpy as np

def enrichir_donnees(df):
    """
    Crée des interactions entre variables explicatives (Feature Engineering).
    """
    df = df.copy()
    
    # --- 1. CONVERSIONS PRÉLIMINAIRES (pour la physique) ---
    # Conversion Température Kelvin -> Celsius (si t > 200, c'est du Kelvin)
    if df['t'].mean() > 200:
        df['temp_c'] = df['t'] - 273.15
        df['temp_max_c'] = df['tx12'] - 273.15
        df['temp_min_c'] = df['tn12'] - 273.15
    else:
        df['temp_c'] = df['t']
        df['temp_max_c'] = df['tx12']
        df['temp_min_c'] = df['tn12']

    # --- 2. INTERACTIONS MÉTÉO (Facteurs de propagation) ---
    # Amplitude thermique journalière (signe de changement de temps)
    df['meteo_amplitude_thermique'] = df['temp_max_c'] - df['temp_min_c']
    
    # "Wind Chill" proxy : Le froid ressenti est pire avec du vent
    # Plus c'est froid (négatif) et venteux, plus l'indice est bas/fort
    df['meteo_froid_vent'] = df['temp_c'] * df['ff']
    
    # Indice de condition favorable au virus : Froid + Humide
    # (On inverse la temp pour que "plus grand" = "plus froid")
    df['meteo_froid_humide'] = (20 - df['temp_c']) * df['u'] / 100
    
    # Point de rosée dépression (T - Td) : indique la saturation de l'air
    # Une faible valeur signifie un air proche de la saturation (brouillard/humide)
    if 'td' in df.columns:
        df['meteo_point_rosee_diff'] = df['t'] - df['td']

    # --- 3. INTERACTIONS GOOGLE (Signaux sociaux) ---
    # Normalisation des requêtes (0-100 -> 0-1)
    if 'requete_grippe' in df.columns:
        google_norm = df['requete_grippe'] / 100.0
        
        # Interaction Google x Densité de population
        # (Beaucoup de recherches dans une zone très peuplée = risque élevé)
        if 'pop_total' in df.columns:
            df['google_x_pop'] = google_norm * np.log1p(df['pop_total'])
            
        # Interaction Google x Météo (Recherches augmentées par le froid ?)
        df['google_x_froid'] = google_norm * (20 - df['temp_c'])

    # --- 4. INTERACTIONS DÉMOGRAPHIQUES (Vulnérabilité) ---
    if 'pop_total' in df.columns:
        # Part des populations vulnérables
        if 'pop_60_74' in df.columns and 'pop_75_plus' in df.columns:
            df['demo_pct_vieux'] = (df['pop_60_74'] + df['pop_75_plus']) / df['pop_total']
        
        if 'pop_0_19' in df.columns:
            df['demo_pct_jeunes'] = df['pop_0_19'] / df['pop_total']
            
        # Densité approximative (Population / const ou juste Pop relative)
        # Utile pour les modèles basés sur les arbres
        df['demo_densite_proxy'] = np.log1p(df['pop_total'])

    # --- 5. INTERACTIONS TEMPORELLES (Cyclicité) ---
    # Numéro de semaine (déjà extrait souvent, mais utile pour le croisement)
    # Sinus/Cosinus déjà gérés dans votre script précédent, mais on peut ajouter:
    # Saison One-Hot numérique simple (si 'saison' est textuel)
    season_map = {'Hiver': 4, 'Automne': 3, 'Printemps': 2, 'Ete': 1}
    if 'saison' in df.columns:
        df['saison_num'] = df['saison'].map(season_map).fillna(0)
        # Croisement Froid x Saison
        df['froid_x_saison'] = df['temp_c'] * df['saison_num']

    return df

# --- UTILISATION ---

# Chargement 
train = pd.read_csv('train_full.csv')
test = pd.read_csv('test_full.csv')

print("Génération des interactions...")
X_train_enriched = enrichir_donnees(train)
X_test_enriched = enrichir_donnees(test)

# Vérification des nouvelles colonnes
new_cols = [c for c in X_train_enriched.columns if c not in train.columns]
print(f"Nouvelles colonnes créées ({len(new_cols)}) :")
print(new_cols)

print("\nAperçu des interactions :")
print(X_train_enriched[new_cols].head())


X_train_enriched.to_csv('train_enriched.csv', index=False)
X_test_enriched.to_csv('test_enriched.csv', index=False)
