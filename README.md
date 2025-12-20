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
│   ├── raw/                               # Données brutes (non transformées)
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
│   │   │                                  
│   │   ├── pop_train.csv                  # Données démographiques normalisées (train)
│   │   ├── pop_test.csv                   # Données démographiques normalisées (test)
│   │   └── train_finale.csv               # Dataset final :  concaténation démographie + Google Trends + train
│   │                                     
│
├── notebooks/                             
│   ├── 01_EDA_test_melina.ipynb            # EDA du test (travail de Melina)
│   ├── 01_preprocessing_train.ipynb        # Merge démographie + météorologie
│   ├── 011_preprocessing_demographique.ipynb  #Préprocessing complet pour obtenir : train final (démographie + requêtes + train)
│   ├── 02_EDA_demographique_pop.ipynb      # EDA donnees demographiques+pop avec train
│   ├── 03_EDA_donneesmeteo.ipynb           # EDA donnees meteo avec train
│   ├── 04_EDA_final.ipynb                  # EDA fusion avec toutes les donnees (meteo, demographiques, pop et train)  feature engeneering et choix du modele
│   ├── 05_notebook_final.ipynb             # feature engeneering et modelisation
│
├── src/                                  
│   ├── __pycache__/                       # Cache Python
│   ├── preprocessing.py                  # Fonctions utilitaires utilisées : dans les notebooks de preprocessing                                    
│
├── results/                               # Résultats du modèle
│   ├── submissions/                
│   │   └── sample_submission.csv          # Fichiers de soumission du prof
│
├── reports/                               # Rapports et présentations
│   ├── rapport_final.pdf
│   └── presentation.pptx
│
├── docs/                                  
│   └── doc_data_StationMeteo.pdf
│
├── pyproject.toml                         # Dépendances et configuration du projet
├── uv.lock                                # Lockfile des dépendances
└── README.md                              # Documentation principale

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





