# (Data Understanding)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# charger le fichier
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ObesityDataSet_raw_and_data_sinthetic.csv'))

# afficher les premières lignes
print(df.head())

# vérifier les valeurs manquantes
print(df.isnull().sum())

# Vérifier les statistiques générales
df.describe()

# Vérifier les doublons
df.duplicated().sum()

# supprimer les lignes dupliquées
df = df.drop_duplicates()

# réinitialiser l'index
df = df.reset_index(drop=True)

# détecter colonnes catégorielles
col_category = df.select_dtypes(include=['object']).columns

# transformer en category
df[col_category] = df[col_category].astype('category')

# vérifier
df.info()
# yes/no variables
yes_no_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']

for col in yes_no_cols:
    df[col] = df[col].str.lower()
    df[col] = df[col].map({'yes': 1, 'no': 0})
df['NObeyesdad'] = df['NObeyesdad'].astype('category').cat.codes
df['Gender']     = df['Gender'].map({'Male': 0, 'Female': 1})
df['CAEC']       = df['CAEC'].apply(lambda x: 3 if x == 'Always' else (2 if x == 'Frequently' else (1 if x == 'Sometimes' else 0)))
df['CALC']       = df['CALC'].apply(lambda x: 3 if x == 'Always' else (2 if x == 'Frequently' else (1 if x == 'Sometimes' else 0)))
df['MTRANS']     = df['MTRANS'].map({'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4})