# Projet-Kaggle


# 🦠 Prédiction du Taux de Grippe - Projet Machine Learning

Prédiction hebdomadaire du taux de grippe par région française à partir de données météo, démographiques et Google Analytics.

## 📂 Structure du projet
```
flu-prediction/
│
├── data/                          # Données brutes (non versionnées)
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── sample_submission.csv
│   │   ├── ListedesStationsMeteo.csv
│   │   ├── DonneesMeteorologiques/
│   │   │   └── synop.YYYYWW.csv (multiples fichiers)
│   │   ├── RequetesGoogleParRegion/
│   │   │   └── *.csv (22 fichiers)
│   │   └── estim-pop-areg-sexe-gca-1975-2015.xls
│   │
│   └── processed/                 # Données transformées
│       ├── meteo_hebdo_region.csv
│       ├── google_hebdo_region.csv
│       ├── demo_hebdo_region.csv
│       ├── features_temporelles.csv
│       ├── train_enriched.csv
│       └── test_enriched.csv
│
├── notebooks/                     # Notebooks Jupyter
│   ├── 01_EDA_train_test.ipynb           # Exploration données principales
│   ├── 02_EDA_meteo.ipynb                # Analyse données météo
│   ├── 03_EDA_google.ipynb               # Analyse Google Analytics
│   ├── 04_EDA_demographie.ipynb          # Analyse démographie
│   ├── 05_integration_donnees.ipynb      # Merge de toutes les sources
│   ├── 06_baseline_models.ipynb          # Modèles de référence
│   ├── 07_modeling_ML.ipynb              # Modèles ML avancés
│   └── 08_final_predictions.ipynb        # Génération submission finale
│
├── src/                           # Code source Python
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── load_meteo.py         # Chargement et agrégation météo
│   │   ├── load_google.py        # Chargement et transformation Google
│   │   ├── load_demo.py          # Chargement et interpolation démographie
│   │   └── merge_data.py         # Pipeline d'intégration
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── temporal_features.py  # Features temporelles
│   │   └── lag_features.py       # Features retardées
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py           # Modèles baseline
│   │   └── ml_models.py          # Modèles ML (RF, XGBoost...)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # Calcul RMSE et autres métriques
│       └── visualization.py      # Fonctions de visualisation
│
├── results/                       # Résultats et soumissions
│   ├── submissions/
│   │   ├── submission_baseline.csv
│   │   ├── submission_v1.csv
│   │   └── submission_final.csv
│   │
│   └── model_comparison.csv       # Comparaison des performances
│
├── reports/                       # Rapports et présentations
│   ├── rapport_final.pdf
│   └── presentation.pptx
│
├── docs/                          # Documentation
│   └── doc_data_StationMeteo.pdf
│
├── requirements.txt               # Dépendances Python
├── .gitignore
└── README.md
```

## 🚀 Installation
```bash
# Cloner le repository
git clone https://github.com/votre-equipe/flu-prediction.git
cd flu-prediction

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Utilisation

### 1. Préparation des données
```bash
# Placer les données brutes dans data/raw/
# Exécuter les notebooks d'EDA (01 à 04)
# Exécuter le notebook d'intégration (05)
```

### 2. Modélisation
```bash
# Baseline : notebook 06
# ML avancé : notebook 07
```

### 3. Génération de la soumission
```bash
# Notebook 08 : génère results/submissions/submission_final.csv
```

## 👥 Équipe

- **Personne 1** : Données météo + EDA principal
- **Personne 2** : Google Analytics
- **Personne 3** : Démographie + features temporelles
- **Personne 4** : Intégration + baseline + coordination

## 📈 Résultats

- RMSE baseline : [À compléter]
- RMSE meilleur modèle : [À compléter]
- Classement Kaggle : [À compléter]

## 📝 Notes

- Variable cible : `TauxGrippe` (taux pour 100 000 habitants)
- Période : 2004-2016
- Granularité : hebdomadaire par région
- 22 régions françaises

## 🔗 Liens utiles

- [Challenge Kaggle](#)
- [Documentation INSEE](http://www.insee.fr)
- [Google Trends](https://trends.google.com)
```

---

## ⚙️ Fichier .gitignore suggéré
```
# Données (trop volumineuses)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Notebooks checkpoints
.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Résultats temporaires
*.log
*.tmp
