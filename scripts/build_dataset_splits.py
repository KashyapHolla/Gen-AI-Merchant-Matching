#%% 

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

#%%
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"

REGISTRY_FILE   = DATA / "Business_Registry.csv"
TX_FILE         = DATA / "Synthetic_Transactions.csv"
TRAIN_OUT_FILE  = DATA / "tx_train.csv"
TEST_OUT_FILE   = DATA / "tx_test.csv"

#%%
print("Loading datasets…")
registry_df = pd.read_csv(REGISTRY_FILE)
transaction_df = pd.read_csv(TX_FILE)

#%%
tx_train, tx_test = train_test_split(
        transaction_df,
        test_size   = 0.10,
        random_state= 42,      # reproducible
        shuffle     = True
)

print(f"Train rows: {len(tx_train)} | Test rows: {len(tx_test)}")

#%%
tx_train.to_csv(TRAIN_OUT_FILE, index=False)
tx_test.to_csv(TEST_OUT_FILE,  index=False)

print("Saved:")
print("Dataset split complete.")

# %%
