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
df.to_csv("data_clean.csv", index=False)
df.info()
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

df.to_csv("data_clean.csv", index=False)
df.info()

correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(11, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x=df["NObeyesdad"])

labels = [
    "Insufficient Weight",
    "Normal Weight",
    "Overweight I",
    "Overweight II",
    "Obesity I",
    "Obesity II",
    "Obesity III"
]

plt.xticks(ticks=range(len(labels)), labels=labels, rotation=30)
plt.title("Distribution of Obesity Classes")
plt.xlabel("Obesity Level")
plt.ylabel("Count")
plt.subplots_adjust(bottom=0.25)
# ─────────────────────────────────────────────────────────────────────────────
#  FONCTION : optimize_memory  (version corrigée + enrichie)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_memory(df, category_threshold=0.5, verbose=True):
    """
    Réduit la mémoire d'un DataFrame pandas en optimisant les types de données.

    Améliorations par rapport à la version originale
    ─────────────────────────────────────────────────
    1. Colonnes OBJECT → category quand le ratio unique/total est faible.
       C'est souvent le gain le plus important (ignoré dans l'ancienne version).
    2. Colonnes déjà CATEGORY → vérification et conservation.
    3. Rapport détaillé colonne par colonne (optionnel via verbose).

    Paramètres
    ──────────
    df                 : DataFrame à optimiser (modifié sur une copie)
    category_threshold : seuil de cardinalité relative au-dessous duquel
                         une colonne object est convertie en category.
                         Ex : 0.5 → si nb_valeurs_uniques / nb_lignes < 0.5
                         (par défaut 0.5 — couvre la quasi-totalité des cas)
    verbose            : affiche le rapport colonne par colonne si True

    Retourne
    ────────
    DataFrame optimisé (copie — le DataFrame original n'est pas modifié)
    """

    df = df.copy()
     start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"\n{'─'*55}")
    print(f"  Mémoire AVANT optimisation : {start_mem:.4f} MB")
    print(f"{'─'*55}")

    report = []

    for col in df.columns:
        col_type  = df[col].dtype
        before_mb = df[col].memory_usage(deep=True) / 1024 ** 2
        converted = False

        # ── 1. Colonnes object ────────────────────────────────
        if col_type == object:
            n_unique = df[col].nunique()
            ratio    = n_unique / len(df)

            if ratio < category_threshold:
                # Faible cardinalité → category (gain mémoire maximal)
                df[col]   = df[col].astype('category')
                converted = True
            # Sinon on laisse en object (haute cardinalité = texte libre)

        # ── 2. Colonnes category (déjà converties) ───────────
        elif str(col_type) == 'category':
            # Déjà optimal, rien à faire
            pass

        # ── 3. Colonnes numériques entières ──────────────────
        elif str(col_type)[:3] == 'int':
            c_min, c_max = df[col].min(), df[col].max()

            if   c_min >= np.iinfo(np.int8).min  and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
            converted = True

        # ── 4. Colonnes numériques flottantes ─────────────────
        elif str(col_type)[:5] == 'float':
            c_min, c_max = df[col].min(), df[col].max()

            # Note : float16 perd beaucoup de précision → on commence à float32
            if   c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
            converted = True

        after_mb = df[col].memory_usage(deep=True) / 1024 ** 2

        if verbose:
            arrow   = "→" if converted else " "
            gain    = before_mb - after_mb
            gain_pct = (gain / before_mb * 100) if before_mb > 0 else 0
            report.append({
                "Colonne":    col,
                "Type avant": str(col_type),
                "Type après": str(df[col].dtype),
                "Avant (KB)": round(before_mb * 1024, 2),
                "Après (KB)": round(after_mb  * 1024, 2),
                "Gain %":     f"{gain_pct:.1f}%",
            })

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    print(f"\n  Mémoire APRÈS optimisation : {end_mem:.4f} MB")
    print(f"  Réduction totale           : {(start_mem - end_mem):.4f} MB  "
          f"({100 * (start_mem - end_mem) / start_mem:.1f}%)")
    print(f"{'─'*55}\n")

    if verbose and report:
        report_df = pd.DataFrame(report)
        print(report_df.to_string(index=False))
        print()

    return df
