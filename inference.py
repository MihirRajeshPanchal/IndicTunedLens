import torch
import pandas as pd
import numpy as np
import os
import json
import argparse
from tuned_lens.nn.lenses import TunedLens, LogitLens
from tuned_lens.nn.unembed import Unembed
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLensConfig
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Run Tuned Lens analysis on CSV prompts')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name or path (e.g., multi-reason/models/sarvam-1)')
    parser.add_argument('--lens_path', type=str, required=True,
                        help='Path to tuned lens folder containing config.json and params.pt')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to input CSV file with questions and options')
    parser.add_argument('--output_dir', type=str, default='tunedlens/results',
                        help='Directory to save output CSV files (default: tunedlens/results)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of top predictions to extract (default: 10)')
    parser.add_argument('--head', type=int, default=5,
                        help='Number of prompts to process from CSV (0 for all, default: 5)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cpu, etc.). Auto-detect if not specified')
    return parser.parse_args()

def load_model_and_tokenizer(model_name, device):
    print(f"Loading model: {model_name}")
    
    # Extract GPU number from device if it's a CUDA device
    if 'cuda' in str(device):
        gpu_id = int(str(device).split(':')[1]) if ':' in str(device) else 0
        device_map = {"": gpu_id}
    else:
        device_map = 'auto'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
def load_tuned_lens(lens_path, model, device):
    print(f"Loading tuned lens from: {lens_path}")
    config_path = os.path.join(lens_path, "config.json")
    params_path = os.path.join(lens_path, "params.pt")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    fixed_config = {
        "base_model_name_or_path": config_dict.get("base_model_name_or_path", model.config._name_or_path),
        "d_model": config_dict.get("d_model", model.config.hidden_size),
        "num_hidden_layers": config_dict.get("num_layers", model.config.num_hidden_layers),
        "bias": config_dict.get("bias", True),
        "base_model_revision": config_dict.get("base_model_revision", None),
        "unembed_hash": config_dict.get("unembed_hash", None),
        "lens_type": config_dict.get("lens_type", "linear_tuned_lens")
    }
    
    config = TunedLensConfig.from_dict(fixed_config)
    unembed = Unembed(model).to(device)
    tuned_lens = TunedLens(unembed, config).to(device)
    state_dict = torch.load(params_path, map_location=device)
    
    layer_translator_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("layer_translators."):
            new_key = key.replace("layer_translators.", "")
            layer_translator_state_dict[new_key] = value
    
    layer_indices = set()
    for key in layer_translator_state_dict.keys():
        if '.' in key:
            layer_idx = int(key.split('.')[0])
            layer_indices.add(layer_idx)
    
    max_layer_idx = max(layer_indices) if layer_indices else 0
    actual_num_translators = max_layer_idx + 1
    
    if actual_num_translators != len(tuned_lens.layer_translators):
        fixed_config["num_hidden_layers"] = actual_num_translators
        config = TunedLensConfig.from_dict(fixed_config)
        tuned_lens = TunedLens(unembed, config).to(device)
    
    tuned_lens.layer_translators.load_state_dict(layer_translator_state_dict)
    
    original_forward = tuned_lens.forward
    
    def patched_forward(self, hidden_states, layer_idx):
        if layer_idx < len(self.layer_translators):
            return original_forward(hidden_states, layer_idx)
        else:
            return self.unembed(hidden_states)
    
    tuned_lens.forward = patched_forward.__get__(tuned_lens, TunedLens)
    
    return tuned_lens

def create_prompt(question, options):
    prompt = f"Question: {question}\nOptions:\n"
    for i, option in enumerate(options, 1):
        prompt += f"{i}. {option}\n"
    prompt += "Please select the correct option."
    print(f"\nCreated prompt: {prompt}")
    return prompt

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def extract_layer_logits_and_probabilities(lens, text, model, tokenizer, device, k=10):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    with torch.no_grad():
        model_outputs = model(input_ids, output_hidden_states=True)
        hidden_states = model_outputs.hidden_states
    
    results = []
    
    for layer_idx in range(len(hidden_states)):
        layer_hidden = hidden_states[layer_idx].squeeze(0)
        
        for token_pos in range(layer_hidden.shape[0]):
            token_hidden = layer_hidden[token_pos]
            
            try:
                if layer_idx < len(lens.layer_translators) if hasattr(lens, 'layer_translators') else True:
                    logits = lens.forward(token_hidden, layer_idx)
                else:
                    logits = lens.unembed(token_hidden)
                
                if logits.dim() > 1:
                    logits = logits.squeeze()
                
                probs = torch.softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, k)
                
                top_k_tokens = []
                for idx in top_k_indices:
                    try:
                        token_text = tokenizer.decode([idx.item()], skip_special_tokens=False)
                        token_text = token_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        top_k_tokens.append(token_text)
                    except:
                        top_k_tokens.append(f"<TOKEN_{idx.item()}>")
                
                for rank in range(k):
                    token_id = top_k_indices[rank].item()
                    input_token_clean = tokens[token_pos] if token_pos < len(tokens) else '<pad>'
                    input_token_clean = input_token_clean.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    
                    predicted_token = top_k_tokens[rank]
                    language = detect_language(predicted_token)
                    
                    results.append({
                        'layer': layer_idx,
                        'token_position': token_pos,
                        'input_token': input_token_clean,
                        'rank': rank + 1,
                        'predicted_token': predicted_token,
                        'predicted_token_id': token_id,
                        'logit': float(logits[token_id].item()),
                        'probability': float(top_k_probs[rank].item()),
                        'language': language
                    })
                    
            except Exception as e:
                print(f"Error processing layer {layer_idx}, position {token_pos}: {e}")
                continue
    
    return pd.DataFrame(results)

def save_logits_to_csv(lens, text, lens_name, output_dir, model, tokenizer, device, k=10, prompt_id=None):
    df = extract_layer_logits_and_probabilities(lens, text, model, tokenizer, device, k=k)
    
    if prompt_id is not None:
        df['prompt_id'] = prompt_id
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{lens_name.lower()}_logits_probabilities{'_' + str(prompt_id) if prompt_id is not None else ''}.csv")
    df.to_csv(filename, index=False, escapechar='\\', quoting=1, encoding='utf-8')
    
    return df

def process_csv_prompts(csv_path, tuned_lens, logit_lens, model, tokenizer, device, output_dir, k=10, head=5):
    try:
        df_input = pd.read_csv(csv_path, encoding='utf-8')
        df_input = df_input.head(head) if head > 0 else df_input
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return
    
    required_columns = ['instruction', 'options']
    if not all(col in df_input.columns for col in required_columns):
        print(f"Error: CSV file must contain columns: {required_columns}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    tuned_dfs = []
    logit_dfs = []
    
    for idx, row in df_input.iterrows():
        prompt_id = idx
        try:
            options = eval(row['options']) if isinstance(row['options'], str) else row['options']
            if not isinstance(options, (list, tuple)):
                print(f"Skipping invalid options format at index {prompt_id}")
                continue
        except:
            print(f"Skipping invalid options format at index {prompt_id}")
            continue
            
        instruction = str(row['instruction'])
        text = create_prompt(instruction, options)
        
        if not text or text.strip() == '':
            print(f"Skipping empty prompt at index {prompt_id}")
            continue
            
        df_tuned = save_logits_to_csv(tuned_lens, text, "TunedLens", output_dir, model, tokenizer, device, k=k, prompt_id=prompt_id)
        tuned_dfs.append(df_tuned)
        
        df_logit = save_logits_to_csv(logit_lens, text, "LogitLens", output_dir, model, tokenizer, device, k=k, prompt_id=prompt_id)
        logit_dfs.append(df_logit)
    
    if tuned_dfs:
        combined_tuned = pd.concat(tuned_dfs, ignore_index=True)
        combined_tuned_filename = os.path.join(output_dir, "tunedlens_combined_logits_probabilities.csv")
        combined_tuned.to_csv(combined_tuned_filename, index=False, escapechar='\\', quoting=1, encoding='utf-8')
        print(f"Saved combined TunedLens results to: {combined_tuned_filename}")
    
    if logit_dfs:
        combined_logit = pd.concat(logit_dfs, ignore_index=True)
        combined_logit_filename = os.path.join(output_dir, "logitlens_combined_logits_probabilities.csv")
        combined_logit.to_csv(combined_logit_filename, index=False, escapechar='\\', quoting=1, encoding='utf-8')
        print(f"Saved combined LogitLens results to: {combined_logit_filename}")

def main():
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    
    # Load tuned lens
    tuned_lens = load_tuned_lens(args.lens_path, model, device)
    
    # Load logit lens
    print("Loading LogitLens...")
    logit_lens = LogitLens.from_model(model).to(device)
    
    # Process CSV prompts
    print(f"\nProcessing CSV file: {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top-k predictions: {args.k}")
    print(f"Number of prompts to process: {args.head if args.head > 0 else 'all'}\n")
    
    process_csv_prompts(
        args.csv_path,
        tuned_lens,
        logit_lens,
        model,
        tokenizer,
        device,
        args.output_dir,
        k=args.k,
        head=args.head
    )
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()