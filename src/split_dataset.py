import os
import pandas as pd
import numpy as np
import json
import base64
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import glob

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TONGUEEXPERT_DIR = os.path.join(DATA_DIR, "TonguExpertDatabase")
RAW_IMAGES_DIR = os.path.join(TONGUEEXPERT_DIR, "TongueImage", "Raw")
LABELS_FILE = os.path.join(TONGUEEXPERT_DIR, "Phenotypes", "L2_Labels_Predict.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")
TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_test_and_train_split():
    """Create test and train splits from the original dataset."""
    # Check if the data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Read the label file
    print(f"Reading labels from {LABELS_FILE}")
    df = pd.read_csv(LABELS_FILE, sep='\t')
    
    # Process L2_Labels_Predict.txt: Convert "None" to "NaN" in fissure_label and tooth_mk_label columns
    for col in ['fissure_label', 'tooth_mk_label']:
        df[col] = df[col].replace("None", "NaN")
        df[col] = df[col].fillna("NaN")  # Also handle actual NaN values
    
    # Print dataset statistics
    print(f"Total samples: {len(df)}")
    print("\nLabel distribution:")
    for col in ['coating_label', 'tai_label', 'zhi_label', 'fissure_label', 'tooth_mk_label']:
        print(f"\n{col}:")
        counts = df[col].value_counts()
        for label, count in counts.items():
            print(f"  {label}: {count} ({count/len(df)*100:.2f}%)")
    
    # Verify that all image files exist
    print("\nVerifying image files...")
    missing_images = []
    for sid in df['SID']:
        img_path = os.path.join(RAW_IMAGES_DIR, f"{sid}.jpg")
        if not os.path.exists(img_path):
            missing_images.append(sid)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} image files are missing. First 10: {missing_images[:10]}")
        # Remove samples with missing images
        df = df[~df['SID'].isin(missing_images)]
        print(f"Removed {len(missing_images)} samples with missing images. Remaining: {len(df)}")
    else:
        print("All image files exist.")
    
    # For stratification, we'll use a combination of all labels
    # We'll create a composite label to try to balance all categories
    # This is a simplification but helps maintain distribution across categories
    df['composite_label'] = df['coating_label'] + "_" + df['tai_label'] + "_" + df['zhi_label']
    
    # Split into train and test sets
    try:
        # Try stratified split on composite label
        train_df, test_df = train_test_split(
            df, 
            test_size=500,
            random_state=RANDOM_SEED,
            stratify=df['composite_label']
        )
    except ValueError as e:
        print(f"Stratified split on composite label failed: {e}")
        print("Trying stratified split on individual labels...")
        
        # Try stratified split on one of the main labels
        try:
            train_df, test_df = train_test_split(
                df, 
                test_size=500,
                random_state=RANDOM_SEED,
                stratify=df['coating_label']
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Falling back to random split")
            train_df, test_df = train_test_split(
                df, 
                test_size=500,
                random_state=RANDOM_SEED
            )
    
    print(f"\nSplit complete. Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Verify distribution in test set
    print("\nTest set label distribution:")
    for col in ['coating_label', 'tai_label', 'zhi_label', 'fissure_label', 'tooth_mk_label']:
        print(f"\n{col}:")
        train_counts = train_df[col].value_counts()
        test_counts = test_df[col].value_counts()
        
        for label in sorted(set(train_counts.index) | set(test_counts.index)):
            train_count = train_counts.get(label, 0)
            test_count = test_counts.get(label, 0)
            original_count = df[col].value_counts().get(label, 0)
            
            train_pct = train_count / len(train_df) * 100
            test_pct = test_count / len(test_df) * 100
            original_pct = original_count / len(df) * 100
            
            print(f"  {label}: Original: {original_pct:.2f}%, Train: {train_pct:.2f}%, Test: {test_pct:.2f}%")
    
    # Save test set to test.txt
    # For test.txt: Make sure fissure_label and tooth_mk_label have "NaN" instead of "None" or NaN
    for col in ['fissure_label', 'tooth_mk_label']:
        test_df[col] = test_df[col].replace("None", "NaN")
        test_df[col] = test_df[col].fillna("NaN")
    
    # Remove the composite_label column
    if 'composite_label' in test_df.columns:
        test_df = test_df.drop(columns=['composite_label'])
    
    # Save the test file
    test_df.to_csv(TEST_FILE, sep='\t', index=False)
    print(f"\nTest set saved to {TEST_FILE}")
    
    # Create train.jsonl
    print(f"\nCreating {TRAIN_JSONL}...")
    
    # For train.jsonl: Make sure fissure_label and tooth_mk_label have "NaN" instead of "None" or NaN
    for col in ['fissure_label', 'tooth_mk_label']:
        train_df[col] = train_df[col].replace("None", "NaN")
        train_df[col] = train_df[col].fillna("NaN")
    
    with open(TRAIN_JSONL, 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            sid = row['SID']
            image_path = os.path.join(RAW_IMAGES_DIR, f"{sid}.jpg")
            
            if os.path.exists(image_path):
                # Create the jsonl entry with proper handling of NaN values
                entry = {
                    "messages": [
                        {
                            "role": "system", 
                            "content": [
                                {
                                    "text": "你是一位经验丰富的中医（老中医），尤其擅长舌诊。你的任务是运用你专业的视觉判断标准，仔细分析提供的舌头图像，并根据中医（TCM）的经典视觉特征对舌象进行分类。请严格专注于图像本身的视觉信息，并严格遵循下面提供的标签选项和输出格式要求。最终仅输出JSON对象。"
                                }
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": """请根据中医舌诊的视觉判断标准，仔细分析下图，并对以下五个指标进行分类。对于每个指标，请从下面提供的选项中选择**唯一一个**最符合图像视觉特征的**英文标签**。括号中的中文描述了该英文标签对应的中医视觉标准，请以此作为你这位老中医进行视觉判断的核心依据。

1. **`coating_label` (舌苔的质地视觉特征):**
   选项 (Options): [
   `greasy` (视觉上，苔质颗粒细腻致密、或伴有黏液、显得油亮、不清爽。对应中医【腻苔】的典型视觉),
   `greasy_thick` (视觉上，苔质特征同'greasy'，但明显更厚、更密实、刮之不去感更强。对应中医【厚腻苔】的典型视觉),
   `non_greasy` (视觉上，苔质不具备'greasy'的油腻、黏厚感，可能呈现薄、净、颗粒相对清晰或略干的状态。对应中医视觉上非腻苔，如薄苔、正常苔等的质地)
   ]
2. **`tai_label` (舌苔的主要颜色视觉特征):**
   选项 (Options): [
   `white` (视觉上，舌苔整体呈现清晰的白色。对应中医【白苔】的视觉),
   `light_yellow` (视觉上，舌苔整体呈现淡淡的、浅浅的黄色调。对应中医【淡黄苔/薄黄苔】的视觉),
   `yellow` (视觉上，舌苔整体呈现明显、较深的黄色调。对应中医【黄苔】的视觉)
   ]
3. **`zhi_label` (舌头的本体颜色视觉特征，即舌质颜色):**
   选项 (Options): [
   `regular` (视觉上，舌体颜色是健康的淡红色或鲜活的粉红色。对应中医【淡红舌】的标准视觉),
   `dark` (视觉上，舌体颜色明显深红、暗红、绛红，或呈现明显的紫色、青紫色。对应中医【红绛舌/紫暗舌】等的视觉),
   `light` (视觉上，舌体颜色明显浅淡、发白，缺乏红润光泽，呈"缺血"外观。对应中医【淡白舌】的视觉)
   ]
4. **`fissure_label` (舌面上的裂纹视觉特征):**
   选项 (Options): [
   `NaN` (视觉上，舌面上完全没有裂纹或明显的沟壑。对应中医【无裂纹】的视觉),
   `light` (视觉上，舌面可见少量裂纹，或裂纹形态较浅、较细。对应中医【少许/浅裂纹】的视觉),
   `severe` (视觉上，舌面可见较多裂纹，或裂纹形态明显较深、较粗、范围较广。对应中医【多/深裂纹】的视觉)
   ]
   (如果视觉上完全看不到裂纹，请选择 `NaN`。)
5. **`tooth_mk_label` (舌头边缘的齿痕视觉特征):**
   选项 (Options): [
   `NaN` (视觉上，舌头边缘光滑或形态自然，没有牙齿压迫形成的印痕。对应中医【无齿痕】的视觉),
   `light` (视觉上，舌头边缘可见轻微的、较浅的波浪状压痕。对应中医【轻微/浅齿痕】的视觉),
   `severe` (视觉上，舌头边缘可见非常明显的、较深的波浪状压痕，舌体可能显得胖大。对应中医【明显/深齿痕】的视觉)
   ]
   (如果视觉上完全看不到齿痕，请选择 `NaN`。)

请严格按照老中医的视觉判断标准进行评估。你的整个回答**必须**仅仅是一个JSON对象，其中包含这五个**英文**键（`coating_label`, `tai_label`, `zhi_label`, `fissure_label`, `tooth_mk_label`）和它们对应的、你根据视觉判断所选择的**英文**标签值。确保输出的JSON格式正确，不要包含任何括号中的中文描述或其他解释性文字。

输出格式示例 (Example Format):
```json
{"coating_label": "greasy", "tai_label": "white", "zhi_label": "regular", "fissure_label": "NaN", "tooth_mk_label": "light"}
```"""
                                },
                                {
                                    "image": f"{sid}.jpg"
                                }
                            ]
                        },
                        {
                            "role": "assistant", 
                            "content": [
                                {
                                    "text": json.dumps({
                                        "coating_label": row["coating_label"],
                                        "tai_label": row["tai_label"],
                                        "zhi_label": row["zhi_label"],
                                        "fissure_label": row["fissure_label"],
                                        "tooth_mk_label": row["tooth_mk_label"]
                                    })
                                }
                            ]
                        }
                    ]
                }
                
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Training data saved to {TRAIN_JSONL}")
    print("Dataset splitting complete!")

if __name__ == "__main__":
    create_test_and_train_split() 