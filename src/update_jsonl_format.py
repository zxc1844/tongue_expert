import json
import os

# Path to the original and temporary files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
original_file = os.path.join(ROOT_DIR, 'data', 'train.jsonl')
temp_file = os.path.join(ROOT_DIR, 'data', 'train_new.jsonl')

# Process the JSONL file line by line
with open(original_file, 'r', encoding='utf-8') as infile, open(temp_file, 'w', encoding='utf-8') as outfile:
    for line_number, line in enumerate(infile, 1):
        try:
            # Parse the JSON entry
            entry = json.loads(line)
            
            # Update the user content format by removing "type" field
            user_content = entry['messages'][1]['content']
            for item in user_content:
                if 'type' in item and 'text' in item:
                    # Keep the text field, remove type field
                    text = item['text']
                    item.clear()
                    item['text'] = text
                elif 'type' in item and 'image' in item:
                    # Keep the image field, remove type field
                    image = item['image']
                    item.clear()
                    item['image'] = image
            
            # Write the updated JSON to the new file
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Print progress for every 1000 lines
            if line_number % 1000 == 0:
                print(f"Processed {line_number} lines...")
                
        except Exception as e:
            print(f"Error processing line {line_number}: {e}")

# Replace the original file with the new one
os.replace(temp_file, original_file)
print(f"Successfully updated format in {original_file}")

# Verify the update by checking the first line
with open(original_file, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    entry = json.loads(first_line)
    user_content = entry['messages'][1]['content']
    
    # Check if the format is correct (no "type" fields)
    format_correct = all('type' not in item for item in user_content)
    
    if format_correct:
        print("Verification successful: 'type' field has been removed from all content items.")
        print("Example of the new format:")
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    else:
        print("Verification failed: 'type' field still exists in some content items.") 