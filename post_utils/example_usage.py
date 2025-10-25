from json_cleaning import (
    load_json, save_json, compute_empty_ratios, suggest_features_to_remove, clean_json
)

def main():
    # Load the JSON data from a file
    input_file = "comment_example.json"
    output_file = "reddit_cleaned.json"
    data = load_json(input_file)

    # Compute empty ratios for features
    empty_ratios = compute_empty_ratios(data)

    # Suggest features to remove based on empty ratios
    features_to_remove = suggest_features_to_remove(empty_ratios, threshold=0.8)

    # Clean the JSON data by removing unwanted fields
    cleaned_data = clean_json(data, features_to_remove)

    # Save the cleaned JSON data to a file
    save_json(cleaned_data, output_file)

    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    main()