#%%writefile preprocessing.py

import pandas as pd
import unicodedata
from IPython.display import display
from typing import List

#---------------- Preprocessing principal
def normalize_region_name(text):
    # Sécurité si la valeur n'est pas du texte
    if not isinstance(text, str):
        return str(text)
    # Suppression des accents (décomposition des caractères et encodage ASCII)
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')   
    # Majuscules + Suppression des tirets (-) + Suppression des espaces
    text = text.upper().replace("'","").replace("-", "").replace(" ", "")   
    return text

#---------------- Preprocessing données démographiques

def get_population_data(file_path, start_year, end_year):
    """
    Charge le fichier Excel contenant des données de population par année (feuilles),
    filtre les années et extrait les données régionales.

    Args:
        file_path (str): Chemin vers le fichier Excel.
        start_year (int): Année de début (inclus).
        end_year (int): Année de fin (inclus).

    Returns:
        pd.DataFrame: DataFrame consolidé contenant les données filtrées.
    """ 
    # Chargement de toutes les feuilles du fichier Excel
    # sheet_name=None charge tout dans un dictionnaire
    try:
        dict_sheets = pd.read_excel(file_path, sheet_name=None, header=None)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None

    all_data = []

    # Boucle sur chaque feuille
    for sheet_name, df in dict_sheets.items():
        # On ignore la feuille non-pertinente
        if "savoir" in str(sheet_name).lower():
            continue          
        # Récupération de l'année depuis le nom de la feuille
        # Extraction des chiffres du nom
        year_str = ''.join(filter(str.isdigit, str(sheet_name)))      
        if year_str:
            year = int(year_str)       
            # Filtre sur les années 
            if start_year <= year <= end_year:           
                # Extraction des données : Régions(lignes 5 à 27) et ages (1 à 6)
                # Col 0: Région, Col 1-6: Ensemble par âge + Total
                # On s'assure de ne pas dépasser les dimensions si le fichier change légèrement
                try:
                    df_clean = df.iloc[5:27, [0, 1, 2, 3, 4, 5, 6]].copy()         
                    # On renomme les colonnes
                    df_clean.columns = [
                        'region', 'pop_0_19', 'pop_20_39', 'pop_40_59', 
                        'pop_60_74', 'pop_75_plus', 'pop_total']
                    # Nettoyage
                    df_clean = df_clean.dropna(subset=['region'])
                    df_clean['year'] = year
                    all_data.append(df_clean)
                except IndexError:
                    continue

    # Concaténation et résultat
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Tri par année et par région
        final_df = final_df.sort_values(by=['year', 'region'])
        
        print(f"Aperçu des données extraites ({start_year}-{end_year}) :")
        display(final_df.sample(5))      
        return final_df
    else:
        print(f"Aucune donnée trouvée pour la période {start_year}-{end_year}.")
        return None

#---------------- Preprocessing données Google Trends
def clean_google_trends_data(input_file, output_file):
    """
    Lit le fichier Excel brut de Google Trends, convertit la colonne 'month' en datetime,
    renomme les colonnes spécifiques et sauvegarde le résultat.
    """
    # Chargement de toutes les feuilles
    dict_sheets = pd.read_excel(input_file, sheet_name=None)
    cleaned_sheets = []  

    # Nouveaux noms de colonnes
    new_cols = [
        "requete_grippe",
        "requete_grippe_aviaire_vaccin",
        "requete_grippe_aviaire_vaccin_porcine_porc_H1N1_AH1N1_A_mexicaine_Mexique_pandemie"]

    print(f"Traitement de {len(dict_sheets)} feuilles...")
    for sheet_name, df in dict_sheets.items():
        # Conversion de la colonne 'month' en datetime
        df['month'] = pd.to_datetime(df['Mois'])
        # Renommage des 3 autres colonnes
        # On crée un dictionnaire {ancien_nom : nouveau_nom}
        rename_map = {
            df.columns[1]: new_cols[0],
            df.columns[2]: new_cols[1],
            df.columns[3]: new_cols[2]}
        df = df.rename(columns=rename_map)        
        cleaned_sheets.append((sheet_name, df))

    # Sauvegarde dans un nouveau fichier Excel
    if cleaned_sheets:
        with pd.ExcelWriter(output_file) as writer:
            for name, df in cleaned_sheets:
                df.to_excel(writer, sheet_name=name, index=False)
        print(f"Succès ! Fichier nettoyé créé : {output_file}")
    else:
        print("Aucune donnée n'a été traitée.")

####
def consolidate_trends_to_single_sheet(input_file, output_file):
    """
    Lit un fichier Excel contenant plusieurs feuilles (une par région),
    et les consolide en une seule feuille avec une colonne 'region_normalized'.
    """
    print(f"Lecture du fichier : {input_file}")
    try:
        # Chargement de toutes les feuilles dans un dictionnaire
        dict_sheets = pd.read_excel(input_file, sheet_name=None)
    except Exception as e:
        print(f"Erreur lors de la lecture : {e}")
        return None
    
    all_data = []
    print(f"Consolidation de {len(dict_sheets)} feuilles...")
    
    for sheet_name, df in dict_sheets.items():
        # On crée une copie pour ne pas modifier l'original 
        df_temp = df.copy()        
        # Ajout de la colonne région basée sur le nom de la feuille
        df_temp['region_normalized'] = sheet_name        
        # On ajoute au tableau global
        all_data.append(df_temp)

    # Fusion (concaténation) de toutes les données
    final_df = pd.concat(all_data, ignore_index=True)        
    # Réorganisation des colonnes pour avoir 'month' et 'region' au début
    cols = list(final_df.columns) # On récupère la liste des colonnes
    # On s'assure que 'month' et 'region_normalized' sont en premier
    if 'month' in cols and 'region_normalized' in cols:
        # On enlève ces deux colonnes de la liste
        cols.remove('month')
        cols.remove('region_normalized')
        # On recrée la liste dans l'ordre voulu
        new_order = ['month', 'region_normalized'] + cols
        final_df = final_df[new_order]

    # Sauvegarde
    final_df.to_excel(output_file, index=False)
    print(f"Succès ! Fichier unique créé : {output_file}")
    print(f"Dimensions finales : {final_df.shape}")
    return final_df

'''
#----------------
Path = "../data/raw/DonneesMeteorologiques"   # Pour simplifier l'annotation de type ## Roland a rajouté ceci
# Preprocessing données météo 
def split_meteo_files(
    files: List[Path],
    start_year: int,
    end_year: int
) -> List[Path]:
    """
    Sépare les fichiers météo synop.*.csv selon une plage d'années.

    Les fichiers doivent être nommés sous la forme : synop.YYYYMM.csv

    Parameters
    ----------
    files : list of Path
        Liste des chemins vers les fichiers météo.
    start_year : int
        Année de début (incluse).
    end_year : int
        Année de fin (incluse).

    Returns
    -------
    list of Path
        Liste des fichiers correspondant à la plage temporelle.
    """
    selected_files = []

    for f in files:
        # synop.YYYYMM.csv → YYYY
        year = int(f.stem.split(".")[1][:4])

        if start_year <= year <= end_year:
            selected_files.append(f)

    return selected_files
#----------------
'''