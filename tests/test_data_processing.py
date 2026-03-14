import os
import sys
import pandas as pd
import joblib

# chemin du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')

sys.path.insert(0, src_path)

from data_processing import optimize_memory

data_path = os.path.join(project_root, 'data', 'data_clean.csv')
model_path = os.path.join(project_root, 'best_model.pkl')


# -------------------------------------------------
# TEST 1 : verify missing values
# -------------------------------------------------

def test_missing_values():

    print("\nTEST Missing Values")

    df = pd.read_csv(data_path)

    missing = df.isnull().sum().sum()

    print("Missing values:", missing)

    assert missing == 0


# -------------------------------------------------
# TEST 2 : verify optimize_memory
# -------------------------------------------------

def test_optimize_memory():

    print("\nTEST optimize_memory")

    df = pd.DataFrame({
        "Age":[20,30,40],
        "Height":[1.70,1.80,1.65],
        "Weight":[70,80,60]
    })

    before = df.memory_usage().sum()

    df = optimize_memory(df)

    after = df.memory_usage().sum()

    print("Memory before:", before)
    print("Memory after:", after)

    assert after <= before


# -------------------------------------------------
# TEST 3 : verify model loading
# -------------------------------------------------

def test_model_loading():

    print("\nTEST Model Loading")

    model = joblib.load(model_path)

    print("Model type:", type(model))

    assert model is not None