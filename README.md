## 🦠 Prédiction du Taux de Grippe - Projet Machine Learning

Prédiction hebdomadaire du taux de grippe par région française à partir de données météo, démographiques et Google Analytics.

### 📂 Structure du projet
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
│
│
├── notebooks/                     # Notebooks Jupyter
│   ├── 01_EDA_train_test.ipynb           # Exploration données principales merge avec train
│   ├── 02_EDA_meteo.ipynb                # Analyse données météo merge avec train
│   ├── 03_EDA_google.ipynb               # Analyse Google Analytics merge avec train
│   ├── 04_EDA_demographie.ipynb          # Analyse démographie merge avec train
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
│
├── results/                       # Résultats et soumissions
│   ├── submissions/
│   │   ├── submission_baseline.csv
│   │   ├── submission_v1.csv
│   │   └── submission_final.csv
│   
│
├── reports/                       # Rapports et présentations
│   ├── rapport_final.pdf
│   └── presentation.pptx
│
├── docs/                          # Documentation
│   └── doc_data_StationMeteo.pdf
│
├── pyproject.toml               # Dépendances Python
├── uv.lock
└── README.md
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

#### Installer les dépendances
```bash
pip install -r requirements.txt  # A changer, on travaille avec uv nous
```
