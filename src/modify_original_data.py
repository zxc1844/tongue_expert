import pandas as pd
import os

# Define the path to the original data file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TONGUEEXPERT_DIR = os.path.join(DATA_DIR, "TonguExpertDatabase")
LABELS_FILE = os.path.join(TONGUEEXPERT_DIR, "Phenotypes", "L2_Labels_Predict.txt")

# Read the original data file
print(f"Reading the original data file: {LABELS_FILE}")
df = pd.read_csv(LABELS_FILE, sep='\t')

# Print the count of "None" values in fissure_label and tooth_mk_label columns
for col in ['fissure_label', 'tooth_mk_label']:
    none_count = (df[col] == "None").sum()
    print(f"Number of 'None' values in {col}: {none_count}")

# Replace "None" with "NaN" in fissure_label and tooth_mk_label columns
for col in ['fissure_label', 'tooth_mk_label']:
    df[col] = df[col].replace("None", "NaN")
    df[col] = df[col].fillna("NaN")  # Also handle actual NaN values

# Print the count of "NaN" values in fissure_label and tooth_mk_label columns
for col in ['fissure_label', 'tooth_mk_label']:
    nan_count = (df[col] == "NaN").sum()
    print(f"Number of 'NaN' values in {col} after replacement: {nan_count}")

# Save the modified data back to the original file
df.to_csv(LABELS_FILE, sep='\t', index=False)
print(f"Modified data saved back to: {LABELS_FILE}")

# Verify the changes
df_check = pd.read_csv(LABELS_FILE, sep='\t')
for col in ['fissure_label', 'tooth_mk_label']:
    none_count = (df_check[col] == "None").sum()
    nan_count = (df_check[col] == "NaN").sum()
    print(f"Verification - {col}: 'None' count = {none_count}, 'NaN' count = {nan_count}") 