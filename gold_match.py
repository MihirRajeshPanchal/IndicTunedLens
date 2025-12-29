import pandas as pd
import argparse
import os
import glob
from pathlib import Path
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Add match_gold_answer column to lens results')
    parser.add_argument('--original_csv', type=str, required=True,
                        help='Path to original CSV file with questions, options, and answer_key')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing logitlens and tunedlens result CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save modified CSV files with match_gold_answer column')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name for tokenizer (e.g., sarvamai/sarvam-1)')
    parser.add_argument('--lens_type', type=str, default='both', choices=['logitlens', 'tunedlens', 'both'],
                        help='Process only specific lens type or both (default: both)')
    return parser.parse_args()

def load_tokenizer(model_name):
    """Load tokenizer for the model"""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer

def load_original_data(csv_path):
    """Load original CSV and create mapping from prompt_id to answer options"""
    print(f"Loading original CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Create mapping: prompt_id -> (answer_key, options_list)
    answer_mapping = {}
    for idx, row in df.iterrows():
        try:
            # Parse options
            options = eval(row['options']) if isinstance(row['options'], str) else row['options']
            if not isinstance(options, (list, tuple)):
                print(f"Warning: Invalid options format at index {idx}, skipping")
                continue
            
            answer_key = str(row['answer_key']).strip().lower()
            answer_mapping[idx] = {
                'answer_key': answer_key,
                'options': options
            }
        except Exception as e:
            print(f"Warning: Error processing row {idx}: {e}")
            continue
    
    print(f"Loaded {len(answer_mapping)} prompts from original CSV")
    return answer_mapping

def get_gold_answer_text(answer_key, options):
    """Convert answer_key (a, b, c, d) to corresponding option text"""
    key_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    
    answer_key = answer_key.lower().strip()
    if answer_key not in key_to_idx:
        return None
    
    idx = key_to_idx[answer_key]
    if idx >= len(options):
        return None
    
    return options[idx]

def tokenize_and_compare(predicted_token, gold_answer_text, tokenizer):
    """Tokenize both predicted and gold answer, then compare"""
    if gold_answer_text is None or predicted_token is None:
        return False
    
    # Tokenize both strings
    predicted_tokens = tokenizer.tokenize(str(predicted_token))
    gold_tokens = tokenizer.tokenize(str(gold_answer_text))
    
    # Check if predicted token appears in any of the gold answer tokens
    # This handles cases where a single token might be part of the gold answer
    for gold_token in gold_tokens:
        if predicted_token.strip() == gold_token.strip():
            return True
    
    # Also check if the predicted token exactly matches the full gold answer
    if predicted_token.strip() == gold_answer_text.strip():
        return True
    
    return False

def process_lens_file(file_path, answer_mapping, tokenizer, output_dir):
    """Process a single lens result file and add match_gold_answer column"""
    print(f"\nProcessing: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    if 'prompt_id' not in df.columns:
        print(f"Warning: 'prompt_id' column not found in {file_path}, skipping")
        return
    
    # Add match_gold_answer column
    match_results = []
    gold_answer_tokens_list = []
    
    for _, row in df.iterrows():
        prompt_id = row['prompt_id']
        predicted_token = str(row['predicted_token']).strip()
        
        if prompt_id not in answer_mapping:
            match_results.append(None)
            gold_answer_tokens_list.append(None)
            continue
        
        answer_key = answer_mapping[prompt_id]['answer_key']
        options = answer_mapping[prompt_id]['options']
        gold_answer_text = get_gold_answer_text(answer_key, options)
        
        if gold_answer_text is None:
            match_results.append(None)
            gold_answer_tokens_list.append(None)
        else:
            # Tokenize gold answer and store for reference
            gold_tokens = tokenizer.tokenize(str(gold_answer_text))
            gold_answer_tokens_list.append('|'.join(gold_tokens))
            
            # Compare tokenized versions
            match = tokenize_and_compare(predicted_token, gold_answer_text, tokenizer)
            match_results.append(match)
    
    df['gold_answer_tokens'] = gold_answer_tokens_list
    df['match_gold_answer'] = match_results
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_file, index=False, escapechar='\\', quoting=1, encoding='utf-8')
    print(f"Saved to: {output_file}")
    
    # Print statistics
    total_predictions = len(df)
    matches = df['match_gold_answer'].sum() if df['match_gold_answer'].notna().any() else 0
    match_rate = (matches / total_predictions * 100) if total_predictions > 0 else 0
    print(f"Match rate: {matches}/{total_predictions} ({match_rate:.2f}%)")

def main():
    args = parse_args()
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)
    
    # Load original CSV data
    answer_mapping = load_original_data(args.original_csv)
    
    # Determine which lens types to process
    lens_types = []
    if args.lens_type in ['logitlens', 'both']:
        lens_types.append('logitlens')
    if args.lens_type in ['tunedlens', 'both']:
        lens_types.append('tunedlens')
    
    # Process files for each lens type
    for lens_type in lens_types:
        print(f"\n{'='*60}")
        print(f"Processing {lens_type.upper()} files")
        print(f"{'='*60}")
        
        # Find all matching files
        pattern = os.path.join(args.results_dir, f"{lens_type}_logits_probabilities_*.csv")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"No files found matching pattern: {pattern}")
            continue
        
        print(f"Found {len(files)} files to process")
        
        # Process each file
        for file_path in files:
            process_lens_file(file_path, answer_mapping, tokenizer, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Modified files saved to: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()