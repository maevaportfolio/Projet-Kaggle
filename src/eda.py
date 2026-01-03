import pandas as pd



#---------------- EDA : Identification des types de variables
def identify_variable_types(df):
    """
    Identifie et affiche les variables numériques et catégorielles d'un DataFrame.
    Retourne deux listes : (colonnes_numeriques, colonnes_categorielles).
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nVariables numériques ({len(numeric_cols)}) :")
    print("-" * 30)
    if numeric_cols:
        print("\n".join([f" - {col}" for col in numeric_cols]))
    else:
        print(" (Aucune)")

    print(f"\nVariables Catégorielles ({len(categorical_cols)}) :")
    print("-" * 30)
    if categorical_cols:
        print("\n".join([f" - {col}" for col in categorical_cols]))
    else:
        print(" (Aucune)")
        
    return numeric_cols, categorical_cols

# ---------------- Taux et nombre de valeurs manquantes par variable      
def taux_missing_values_nb(df):
    missing_values = df.isnull().mean() * 100 # Calcul du pourcentage de valeurs manquantes par colonne
    missing_values_nb = df.isnull().sum() # Calcul du nombre de valeurs manquantes par colonne
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if missing_values.empty:        
        print("Il n'y a pas de valeurs manquantes dans le dataset.")
    else:
        print("Taux de valeurs manquantes par variable :")
        for col, val in missing_values.items():
            print(f"{col:<10} : {val:.2f}%") 
        print("Nombre de valeurs manquantes par variable :")
        print(missing_values_nb[missing_values_nb > 0].sort_values(ascending=False))