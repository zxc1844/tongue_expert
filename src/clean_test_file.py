import pandas as pd
import os

# Path to test file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_file = os.path.join(ROOT_DIR, 'data', 'test.txt')

# Read the test file
print(f"Reading test file from: {test_file}")
test_df = pd.read_csv(test_file, sep='\t')

# Print initial information
print(f"Initial number of rows: {len(test_df)}")
print(f"Initial columns: {', '.join(test_df.columns)}")

# Remove the composite_label column if it exists
if 'composite_label' in test_df.columns:
    test_df = test_df.drop(columns=['composite_label'])
    print("Removed 'composite_label' column")
else:
    print("'composite_label' column not found")

# Make sure fissure_label and tooth_mk_label have "NaN" instead of "None"
for col in ['fissure_label', 'tooth_mk_label']:
    if col in test_df.columns:
        # Count None values before replacement
        none_count = (test_df[col] == "None").sum()
        print(f"Number of 'None' values in {col}: {none_count}")
        
        # Replace None with NaN
        test_df[col] = test_df[col].replace("None", "NaN")
        test_df[col] = test_df[col].fillna("NaN")
        
        # Count NaN values after replacement
        nan_count = (test_df[col] == "NaN").sum()
        print(f"Number of 'NaN' values in {col} after replacement: {nan_count}")

# Save the cleaned test file
test_df.to_csv(test_file, sep='\t', index=False)
print(f"Cleaned test file saved to: {test_file}")

# Verify the changes
df_check = pd.read_csv(test_file, sep='\t')
print(f"Final number of rows: {len(df_check)}")
print(f"Final columns: {', '.join(df_check.columns)}")
for col in ['fissure_label', 'tooth_mk_label']:
    if col in df_check.columns:
        none_count = (df_check[col] == "None").sum()
        nan_count = (df_check[col] == "NaN").sum()
        print(f"Verification - {col}: 'None' count = {none_count}, 'NaN' count = {nan_count}") 