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
├── data/
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
│   ├── processed/          # Données nettoyées, enrichies et fusionnées
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── train_full.csv
│   │   ├── test_full.csv
│   │   ├── train_with_meteo.csv
│   │   ├── test_with_meteo.csv
│   │   ├── train_meteo_full.csv
│   │   ├── test_meteo_full.csv
│   │   ├── train_meteo_reduit.csv
│   │   ├── test_meteo_reduit.csv
│   │   ├── train_synop_cleaned_full_variables.csv
│   │   ├── train_synop_merged_inner.csv
│   │   ├── test_synop_merged_inner.csv
│   │   ├── pop_train.csv
│   │   ├── pop_test.csv
│   │   ├── google_trends_clean.xlsx
│   │   ├── google_trends_consolide.xlsx
│   │   ├── google_trends_requetes.xlsx
│   │   └── FINAL_TEST.csv
│   │
│   └── data_modelisation/  # Jeux finaux prêts pour l'entraînement et l'inférence
│       ├── train_full.csv  # Jeu d'entraînement final (features complètes)
│       └── test_full.csv   # Jeu de test final (features complètes)
│
├── notebooks/
│   ├── 01_preprocessing_train_test.ipynb
│   ├── 011_preprocessing_pop_requetes.ipynb
│   ├── 012_preprocessing_train_meteo.ipynb
│   ├── 013_preprocessing_test_meteo.ipynb
│   ├── 02_EDA_pop_requetes_meteo.ipynb
│   ├── Modélisation_finale_100.ipynb
│   └── pour_se_guider/     # Notebooks de référence / support pédagogique
│
├── src/
│   ├── preprocessing.py   # Fonctions de préparation des données
│   ├── eda.py              # Fonctions d’analyse exploratoire
│   └── catboost_ancien_meilleur_score.py
│
├── results/
│   ├── img/                # Graphiques et visualisations
│   └── submissions/        # Fichiers de soumission Kaggle
│
├── docs/
│   └── doc_data_StationMeteo.pdf
|
├── reports/                # Rapports et présentations
│   ├── rapport_final.pdf
│   └── presentation.pptx
│
├── deleted_files/          # Archives et anciens scripts (non utilisés)
│
├── README.md
├── pyproject.toml
├── uv.lock
└── .python-version




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













