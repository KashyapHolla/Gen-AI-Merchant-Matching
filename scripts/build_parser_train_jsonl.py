#%%
import pandas as pd
import json
import re
from pathlib import Path
#%%
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"

tx_path = DATA / "tx_train.csv"
registry_path = DATA / "Business_Registry.csv"
output_path = DATA / "parser_train.jsonl"

#%%
# Loading files
tx_df = pd.read_csv(tx_path)
registry_df = pd.read_csv(registry_path).set_index("merchant_id")

#%%

def extract_merchant_id(descriptor: str) -> str | None:
    """Extract merchant ID from descriptor."""
    # Pattern 1: IDs after # symbol (with optional dash)
    match = re.search(r'#\s*-?\s*(\d+)', descriptor)
    if match:
        return match.group(1)
    
    # Pattern 2: IDs in format "xxxx - ##### / LOCATION" 
    match = re.search(r'-\s*(\d+)\s*/\s*', descriptor)
    if match:
        return match.group(1)
    
    # Pattern 3: Long numeric IDs anywhere in string
    match = re.search(r'(\d{10,})', descriptor)
    if match:
        return match.group(1)
        
    # Pattern 4: IDs of 5-6 digits
    match = re.search(r'\b(\d{5,6})\b', descriptor)
    if match:
        return match.group(1)
        
    return None
#%%
# Apply to all descriptors
records = []
skipped = 0

for descriptor in tx_df["messy_descriptor"]:
    try:
        merchant_id = extract_merchant_id(descriptor)
        if merchant_id is None or int(merchant_id) not in registry_df.index:
            skipped += 1
            continue
        
        merchant_id = int(merchant_id)
        row = registry_df.loc[merchant_id].to_dict()
        try:
            target = {
                "brand": row["brand"][merchant_id],
                "merchant_id": merchant_id,
                "city": row["city"][merchant_id],
                "state": row["state"][merchant_id]
            }
        except Exception as e:
            target = {
                "brand": row["brand"],
                "merchant_id": merchant_id,
                "city": row["city"],
                "state": row["state"]
            }

        prompt = f"### User\n{descriptor}\n### Assistant\n"
        completion = json.dumps(target, ensure_ascii=False)

        records.append({
            "prompt": prompt,
            "completion": completion
        })
    except Exception as e:
        print(f"Error processing descriptor: {descriptor}")
        print(f"Error: {e}")
        skipped += 1
# %%
# Save to JSONL
with open(output_path, "w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Created {len(records)} training pairs")
print(f"Skipped {skipped} rows due to missing merchant_id")
print(f"Saved to {output_path.relative_to(ROOT)}")
# %%
