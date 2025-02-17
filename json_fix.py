import json
from pathlib import Path
import logging

def load_json_safely(file_path):
    """
    Safely load JSON data with better error handling and logging
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                # Read the problematic content
                f.seek(0)
                content = f.read()
                logging.error(f"JSON Parse Error at position {e.pos}:")
                logging.error(f"Content snippet: {content[max(0, e.pos-50):min(len(content), e.pos+50)]}")
                raise
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while reading {file_path}: {str(e)}")
        raise

def save_json_safely(data, file_path):
    """
    Safely save JSON data with validation and error handling
    """
    try:
        # Verify data is JSON serializable
        json.dumps(data)

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Verify the saved file
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)

    except TypeError as e:
        logging.error(f"Data is not JSON serializable: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {str(e)}")
        raise

# Modified data processing function
def process_data(data):
    """
    Process and validate the data structure
    """
    if not isinstance(data, (list, dict)):
        raise ValueError(f"Expected list or dict, got {type(data)}")

    if isinstance(data, dict):
        if 'data' not in data:
            data = {'data': data}
    else:
        data = {'data': data}

    return data

# Usage in your notebook:
def save_processed_data(processed_data, output_path="processed_data.json"):
    """
    Save processed data with proper structure and validation
    """
    try:
        data_to_save = process_data(processed_data)
        save_json_safely(data_to_save, output_path)

        # Verify
        loaded_data = load_json_safely(output_path)
        print(f"Dataset saved successfully with {len(loaded_data['data'])} items")
        return True

    except Exception as e:
        logging.error(f"Failed to save processed data: {str(e)}")
        return False
