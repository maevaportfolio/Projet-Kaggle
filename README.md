## 🦠 Prédiction du Taux de Grippe - Projet Machine Learning

Ce projet vise à prédire l’intensité de la grippe par **région** et par **semaine**, en combinant plusieurs sources de données :
- Données épidémiologiques
- Données météorologiques
- Requêtes Google Trends
- Données démographiques

L’objectif est de construire un **dataset final enrichi** pour l’entraînement de modèles de machine learning.

---

### Prérequis
- Python **3.12**
- `uv` comme gestionnaire de dépendances

### 📂 Structure du projet
```
flu-prediction/
│
├── data/                                   # Données du projet
│   ├── raw/                                # Données brutes (non transformées)
│   │   ├── train.csv                        
│   │   ├── test.csv
│   │   ├── ListedesStationsMeteo.csv
│   │   ├── DonneesMeteorologiques/
│   │   │   └── synop.YYYYWW.csv            # Données météo hebdomadaires
│   │   ├── RequetesGoogleParRegion/
│   │   │   └── *.csv                       # 22 fichiers CSV (1 par région)
│   │   └── estim-pop-areg-sexe-gca-1975-2015.xls
│   │
│   ├── processed/                         # Données transformées / finales
│   │   ├── Google_trend_clean.xlsx        # Requêtes Google avec noms de colonnes normalisés
│   │   ├── google_trend_consolidated.xlsx # Consolidation de tous les CSV Google  → 1 fichier Excel, 1 feuille par région → en-têtes non normalisées        
│   │   ├── Google_trends_requetes.xlsx    # Train final Google Trends, 1 seule feuille, fusion de 9 fichiers finaux de requêtes                                                
│   │   ├── pop_train.csv                  # Données démographiques normalisées (train)
│   │   ├── pop_test.csv                   # Données démographiques normalisées (test)
│   │   ├── train_pop_requetes.csv         # Dataset à moitié final :  concaténation démographie + Google Trends + train
│   │   ├── test_pop_requetes.csv          # Dataset à moitié final :  concaténation démographie + Google Trends + test
│   │   ├── train_meteo.csv                # Dataset à moitié final :  concaténation meteo + test  
│   │   ├── test_meteo.csv                 # Dataset à moitié final :  concaténation meteo + test
│   │   ├── train_final.csv                # Dataset final :  concaténation démographie + Google Trends + meteo + train
│   │   └── test_final.csv                 # Dataset final :  concaténation démographie + Google Trends + meteo + test             
│       
├── notebooks/                             
│   ├── 01_EDA_train_melina.ipynb               # EDA du test (travail de Melina) LA PIERRE FONDATRICE
│   ├── 01_preprocessing_train_test.ipynb            # Merge démographie + météorologie 
│   ├──── 011_preprocessing_pop_requetes.ipynb  # Préprocessing complet pour obtenir : train et test final (démographie + requêtes + train)
│   ├──── 012_preprocessing_meteo.ipynb         # Préprocessing complet pour obtenir : train et test final (démographie + meteo)
│   ├── 02_EDA_final.ipynb                      # EDA fusion avec toutes les donnees
│   ├──── 021_EDA_pop_requetes.ipynb            # EDA donnees demographiques + pop avec train
│   ├──── 022_EDA_meteo.ipynb                   # EDA donnees meteo avec train
│   ├── 03_Modélisation.ipynb                   # Modélisations finales
│   ├──── 031_Modélisation_pop_requetes.ipynb   # Modélisations Roland et Maeva
│   └──── 032_Modélisation_meteo.ipynb          # Modélisations MeliNa et Anastasiia
│
├── src/                                  
│   ├── __pycache__/                            # Cache Python
│   ├── preprocessing.py                        # Fonctions utilitaires utilisées : dans les notebooks de preprocessing
│   ├── eda.py                                  # Fonctions utilitaires utilisées : dans les notebooks de l'EDA                                      
│   └── modelisation.py                         # Fonctions utilitaires utilisées : dans les notebooks de modélisation     
|
├── results/                                    # Résultats du modèle
│   ├── submissions/                
│   │   ├── sample_submission.csv               # Fichier de soumission du prof
│   │   ├── sample_submission_naive.csv
│   │   ├── sample_submission_regression_linéaire.csv
│   │   └── sample_submission_random_forest.csv
│   ├── img/                                    # Images des résultats
│   │   ├── .png
│   │   ├── .png
│   │   └── .png
|
├── reports/                                    # Rapports et présentations
│   ├── rapport_final.pdf
│   └── presentation.pptx
│
├── docs/                                  
│   └── doc_data_StationMeteo.pdf
│
├── pyproject.toml                              # Dépendances et configuration du projet
├── uv.lock                                     # Lockfile des dépendances
└── README.md                                   # Documentation principale

```

## 🚀 Installation
```bash
# Cloner le repository
git clone https://github.com/votre-equipe/flu-prediction.git
cd flu-prediction
```

#### Créer environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

```bash
# ou
venv\Scripts\activate  # Windows
```

#### Installer UV


```bash
pip installl uv
```

#### Installer les dépendances
```bash
uv sync --locked
```








