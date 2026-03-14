# (Data Understanding)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
# charger le fichier

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data' , 'ObesityDataSet_raw_and_data_sinthetic.csv'))
df
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
df
# réinitialiser l'index
df = df.reset_index(drop=True)
df
# détecter colonnes catégorielles
col_category = df.select_dtypes(include=['object']).columns

# transformer en category
df[col_category ] = df[col_category ].astype('category')

# vérifier
df.info()
# yes/no variables
yes_no_cols = ['family_history_with_overweight','FAVC','SMOKE','SCC']

for col in yes_no_cols:
    df[col] = df[col].str.lower()
    df[col] = df[col].map({'yes':1,'no':0})
df
df['NObeyesdad'] = df['NObeyesdad'].astype('category').cat.codes
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
df['CAEC'] = df['CAEC'].apply(lambda x: 3 if x == 'Always' else (2 if x == 'Frequently' else (1 if x == 'Sometimes' else 0)))
df['CALC'] = df['CALC'].apply(lambda x: 3 if x == 'Always' else (2 if x == 'Frequently' else (1 if x == 'Sometimes' else 0)))
df['MTRANS'] = df['MTRANS'].map({'Automobile':0, 'Bike':1, 'Motorbike':2, 'Public_Transportation':3, 'Walking':4})
df  
df.to_csv("data_clean.csv", index=False)
df.info()
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
numeric_cols = df.select_dtypes(include=['int64','float64']).columns

plt.figure(figsize=(11,8))

for i, col in enumerate(numeric_cols):
    plt.subplot(4,4,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

plt.show()
plt.figure(figsize=(8,5))
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

plt.show()
def optimize_memory(df):

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage BEFORE optimization: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            df[col] = df[col].astype("category")

        elif col_type != "category":

            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)

                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)

                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            else:

                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    print(f"Memory usage AFTER optimization: {end_mem:.2f} MB")
    print(f"Memory reduced by {(100 * (start_mem - end_mem) / start_mem):.1f}%")

    return df
print("Before optimization:")

df = optimize_memory(df)
df.info(memory_usage="deep")
print("\nAfter optimization:")
df.info(memory_usage="deep")
print("\nAfter optimization:")
df.info(memory_usage="deep")