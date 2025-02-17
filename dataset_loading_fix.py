from datasets import Dataset, load_dataset
import json
import logging

def load_dataset_safely(file_path, field='data'):
    """
    Safely load a JSON file as a Hugging Face dataset with proper error handling
    """
    try:
        # First, safely read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Extract the data field
        if field in raw_data:
            data = raw_data[field]
        else:
            raise KeyError(f"Field '{field}' not found in JSON data")

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(data)

        return dataset

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from {file_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {str(e)}")
        raise

# Modified training data loading function
def load_training_data(file_path="processed_data.json", field='data'):
    """
    Load and prepare training data with validation
    """
    try:
        # Load dataset
        dataset = load_dataset_safely(file_path, field)

        # Validate required fields
        required_fields = ['query_vi', 'response_vi']
        missing_fields = [field for field in required_fields
                         if field not in dataset.features]

        if missing_fields:
            raise ValueError(f"Dataset is missing required fields: {missing_fields}")

        # Shuffle with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=42)

        return dataset

    except Exception as e:
        logging.error(f"Failed to load training data: {str(e)}")
        raise

# Example usage in your training script:
def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)

        # Load the dataset
        data = load_training_data()
        logging.info(f"Successfully loaded dataset with {len(data)} examples")

        # Continue with your training code...

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
