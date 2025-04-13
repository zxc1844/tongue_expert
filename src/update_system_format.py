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
            
            # Update the system message format
            system_message = entry['messages'][0]
            if system_message['role'] == 'system' and isinstance(system_message['content'], str):
                # Convert string content to array with text object
                system_content = system_message['content']
                system_message['content'] = [{"text": system_content}]
            
            # Write the updated JSON to the new file
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Print progress for every 1000 lines
            if line_number % 1000 == 0:
                print(f"Processed {line_number} lines...")
                
        except Exception as e:
            print(f"Error processing line {line_number}: {e}")

# Replace the original file with the new one
os.replace(temp_file, original_file)
print(f"Successfully updated system message format in {original_file}")

# Verify the update by checking the first line
with open(original_file, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    entry = json.loads(first_line)
    system_message = entry['messages'][0]
    
    # Check if the format is correct (system content is now an array with text object)
    if system_message['role'] == 'system' and isinstance(system_message['content'], list):
        print("Verification successful: System message content is now in array format with text object.")
        print("Example of the new system message format:")
        print(json.dumps(system_message, ensure_ascii=False, indent=2))
    else:
        print("Verification failed: System message content is not in the expected format.") 