import json
import os

import jsonlines

def clean_data_by_file_path(input_file_path, output_file_path):
    """
    Reads JSON data, checks if the 'ecg_path' exists, and writes only valid 
    records to a new JSON file.

    Args:
        input_file_path (str): Path to the original JSON file.
        output_file_path (str): Path to save the cleaned JSON file.
    """
    valid_records = []
    deleted_count = 0

    try:
        # 1. Load the data
        with open(input_file_path, 'r') as f:
            data = data = list(jsonlines.Reader(f))
            list(data)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}. Is it valid JSON?")
        return
    
    # Ensure the data is a list of records
    if not isinstance(data, list):
        print("Warning: The loaded JSON data is not a list. Assuming it's a single record.")
        data = [data]

    # 2. Iterate and validate each record
    print(f"Starting validation of {len(data)} records...")
    
    for record in data:
        if "ecg_path" in record:
            ecg_path = record["ecg_path"]
            
            # Use os.path.exists to check if the file exists
            if os.path.exists(ecg_path+".hea"):
                valid_records.append(record)
            else:
                # File not found, delete/skip the record
                print(f"File not found, skipping record with path: {ecg_path}")
                deleted_count += 1
        else:
            # If 'ecg_path' is missing, you may want to keep or delete it.
            # Here, we keep it, assuming missing path means it's still valid data.
            valid_records.append(record) 

    # 3. Save the cleaned data
    with open(output_file_path, 'w') as f:
        # Use indent=4 for human-readable output
        jsonlines.Writer(f).write_all(valid_records)

    print("-" * 30)
    print(f"✅ Validation Complete.")
    print(f"Total records processed: {len(data)}")
    print(f"Records kept: {len(valid_records)}")
    print(f"Records deleted (file not found): {deleted_count}")
    print(f"Cleaned data saved to: {output_file_path}")

# --- Execution ---
input_file = '/content/ecg_llm/ptbxl/ptbxl_ecg_train.jsonl'
output_file = "/content/ecg_llm/ptbxl/ptbxl_ecg_train_cleaned.jsonl"

# Assuming you place your list of records into 'input_data.json'
# If you only have one record, wrap it in a list [record] in the file.
# 



clean_data_by_file_path(input_file, output_file)


input_file2 = '/content/ecg_llm/ptbxl/ptbxl_ecg_test.jsonl'
output_file2 = "/content/ecg_llm/ptbxl/ptbxl_ecg_test_cleaned.jsonl"


clean_data_by_file_path(input_file2, output_file2)



input_file3 = '/content/ecg_llm/ptbxl/ptbxl_ecg_val.jsonl'
output_file3 = "/content/ecg_llm/ptbxl/ptbxl_ecg_val_cleaned.jsonl"


clean_data_by_file_path(input_file3, output_file3)


