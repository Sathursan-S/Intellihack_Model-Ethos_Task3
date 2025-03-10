#!/usr/bin/env python3
import os
import json
import logging
import sys
import argparse

# Setup logging to output in Colab or terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def clean_data(item: dict) -> dict:
    """
    Clean a dictionary by trimming string values and removing keys with empty or null values.
    
    Args:
        item (dict): A dictionary in Alpaca QA format.
        
    Returns:
        dict: Cleaned dictionary. If the dictionary is empty after cleaning, returns {}.
    """
    cleaned = {}
    for key, value in item.items():
        # For strings, trim whitespace
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue  # Skip empty strings
        # Skip None values
        if value is None:
            continue
        cleaned[key] = value
    return cleaned

def combine_json_files(input_dir: str) -> list:
    """
    Read all JSON files in the input directory, combine their contents into a single list,
    and apply cleaning to each item.
    
    Args:
        input_dir (str): Directory containing JSON files.
        
    Returns:
        list: A list of cleaned dictionaries.
    """
    combined_items = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # If data is a list, extend; if a dict, append it.
                    if isinstance(data, list):
                        combined_items.extend(data)
                    elif isinstance(data, dict):
                        combined_items.append(data)
                    else:
                        logger.warning("Unexpected data format in file: %s", filename)
            except Exception as e:
                logger.error("Error reading file %s: %s", filename, str(e))
    
    logger.info("Total items read before cleaning: %d", len(combined_items))
    # Clean each item and filter out empty ones.
    cleaned_items = [clean_data(item) for item in combined_items if clean_data(item)]
    logger.info("Total items after cleaning: %d", len(cleaned_items))
    return cleaned_items

def write_jsonl(data: list, output_file: str):
    """
    Write the list of dictionaries to a JSONL file (one JSON object per line).
    
    Args:
        data (list): List of dictionaries.
        output_file (str): Path to the output JSONL file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as out_f:
            for item in data:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Combined data saved to %s", output_file)
    except Exception as e:
        logger.error("Error writing to output file: %s", str(e))
        sys.exit(1)

def main():
    # parser = argparse.ArgumentParser(
    #     description="Combine all JSON files from an input directory and output a cleaned JSONL file in Alpaca format."
    # )
    # parser.add_argument("--input_dir", type=str, default=".", help="Directory containing JSON files.")
    # parser.add_argument("--output_file", type=str, default="combined_alpaca_dataset.jsonl", help="Output JSONL file path.")
    # args = parser.parse_args()
    
    combined_data = combine_json_files(".")
    write_jsonl(combined_data, "combined_alpaca_dataset.jsonl")
    
if __name__ == "__main__":
    main()
