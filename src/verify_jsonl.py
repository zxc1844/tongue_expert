import json
import os

# Path to the JSONL file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jsonl_file = os.path.join(ROOT_DIR, 'data', 'train.jsonl')

# Read and check the first 10 entries
with open(jsonl_file, 'r', encoding='utf-8') as f:
    for i in range(10):
        line = f.readline()
        if not line:
            break
            
        # Parse the JSON entry
        try:
            entry = json.loads(line)
            
            # Extract the assistant response which contains the labels
            assistant_content = entry['messages'][2]['content'][0]['text']
            labels = json.loads(assistant_content)
            sid = entry['messages'][1]['content'][1]['image'].split('.')[0]
            
            # Print the labels for this entry
            print(f"Entry {i+1} (SID: {sid}):")
            print(f"  fissure_label: {labels['fissure_label']}")
            print(f"  tooth_mk_label: {labels['tooth_mk_label']}")
            print()
            
        except Exception as e:
            print(f"Error processing entry {i+1}: {e}") 