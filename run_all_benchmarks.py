# run_all_benchmarks.py

import os
import json
import pandas as pd
from tqdm import tqdm
import glob
import re
import time

# --- Internal Imports from Text2KGBench ---
# We assume this script is run from the root of the Text2KGBench repo.
# Add src to path to allow imports
import sys
sys.path.append('./src')
from baseline.llm_client import get_llm_client
from evaluation.evaluation import (
    get_triples,
    get_precision,
    get_recall,
    get_f1,
    get_sh,
    get_rh,
    get_oh
)
from baseline.prompt_generation import (
    get_prompt_from_ontology,
    get_prompt,
    get_sentence_similarity
)

# --- Configuration ---
MODELS_TO_RUN = ['llama3.1', 'gemma3', 'gemma3n']
DATASETS = ['wikidata_tekgen', 'dbpedia_webnlg']
# To run a smaller test, set this to True and adjust the limit
RUN_IN_TEST_MODE = False 
TEST_MODE_LIMIT = 5 


def step_1_get_llm_responses(model_name, dataset, test_mode=False):
    """
    Generates prompts and gets responses from the specified LLM.
    """
    print(f"\n--- [Step 1] Getting LLM Responses for model: '{model_name}' on dataset: '{dataset}' ---")
    
    # Initialize the correct client for the model
    try:
        client = get_llm_client(model_name)
    except Exception as e:
        print(f"Could not initialize client for {model_name}. Skipping. Error: {e}")
        return

    # Define paths
    prompt_dir = f'data/baselines/{model_name}/{dataset}/prompts'
    response_dir = f'data/baselines/{model_name}/{dataset}/llm_responses'
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(response_dir, exist_ok=True)

    test_data_path = f'data/{dataset}/test'
    test_files = glob.glob(f'{test_data_path}/*.jsonl')

    for test_file in tqdm(test_files, desc=f"Processing files for {dataset}"):
        file_name = os.path.basename(test_file)
        ontology_name = file_name.replace("_test.jsonl", "")
        
        prompt_file_path = os.path.join(prompt_dir, file_name.replace(".jsonl", "_prompts.jsonl"))
        response_file_path = os.path.join(response_dir, file_name)

        if os.path.exists(response_file_path):
            print(f"Responses for {file_name} already exist. Skipping.")
            continue

        prompts = []
        test_cases = []
        with open(test_file, 'r') as f:
            for line in f:
                test_cases.append(json.loads(line))

        if test_mode:
            test_cases = test_cases[:TEST_MODE_LIMIT]

        for test_case in tqdm(test_cases, desc=f"Generating prompts for {ontology_name}", leave=False):
            # The prompt generation logic is taken from the original notebook
            prompt_context = get_prompt_from_ontology(f'data/{dataset}/ontologies/{ontology_name}_ontology.json')
            train_file_path = f'data/{dataset}/train/{ontology_name}_train.jsonl'
            train_similarity_path = f'data/{dataset}/baselines/test_train_sent_similarity/{ontology_name}_test_train_similarity.jsonl'
            
            demonstrations = get_sentence_similarity(test_case, train_file_path, train_similarity_path)
            
            prompt = get_prompt(prompt_context, demonstrations, test_case['sent'])
            prompts.append({'id': test_case['id'], 'prompt': prompt})
        
        # Save prompts
        with open(prompt_file_path, 'w') as f:
            for p in prompts:
                f.write(json.dumps(p) + '\n')

        # Get and save responses
        with open(response_file_path, 'w') as f:
            for p in tqdm(prompts, desc=f"Getting responses for {ontology_name}", leave=False):
                response = client.get_response(p['prompt'])
                f.write(json.dumps({'id': p['id'], 'llm_response': response}) + '\n')
                # Add a small delay to avoid overwhelming the service
                time.sleep(1) 

    print(f"--- [Step 1] Completed for {model_name} on {dataset} ---")


def step_2_extract_triples(model_name, dataset):
    """
    Extracts triples from the raw LLM responses.
    """
    print(f"\n--- [Step 2] Extracting Triples for model: '{model_name}' on dataset: '{dataset}' ---")
    
    response_dir = f'data/baselines/{model_name}/{dataset}/llm_responses'
    extracted_dir = f'data/baselines/{model_name}/{dataset}/extracted_triples'
    os.makedirs(extracted_dir, exist_ok=True)
    
    response_files = glob.glob(f'{response_dir}/*.jsonl')
    
    if not response_files:
        print(f"No response files found for {model_name} on {dataset}. Skipping.")
        return

    for response_file in tqdm(response_files, desc="Extracting triples"):
        file_name = os.path.basename(response_file)
        extracted_file_path = os.path.join(extracted_dir, file_name)

        if os.path.exists(extracted_file_path):
            print(f"Extracted triples for {file_name} already exist. Skipping.")
            continue
            
        with open(response_file, 'r') as f_in, open(extracted_file_path, 'w') as f_out:
            for line in f_in:
                data = json.loads(line)
                response_text = data.get('llm_response', '')
                
                # This regex is a simple approach from the original notebooks.
                # It might need to be improved for more complex outputs.
                try:
                    triples_str = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL).group(0)
                    triples = json.loads(triples_str)
                except (AttributeError, json.JSONDecodeError):
                    triples = [] # Could not find or parse triples

                f_out.write(json.dumps({'id': data['id'], 'triples': triples}) + '\n')
    
    print(f"--- [Step 2] Completed for {model_name} on {dataset} ---")


def step_3_evaluate_results(model_name, dataset):
    """
    Evaluates the extracted triples against the ground truth.
    """
    print(f"\n--- [Step 3] Evaluating Results for model: '{model_name}' on dataset: '{dataset}' ---")

    extracted_dir = f'data/baselines/{model_name}/{dataset}/extracted_triples'
    eval_dir = f'data/baselines/{model_name}/{dataset}/eval_metrics'
    os.makedirs(eval_dir, exist_ok=True)
    
    extracted_files = glob.glob(f'{extracted_dir}/*.jsonl')

    if not extracted_files:
        print(f"No extracted triple files found for {model_name} on {dataset}. Skipping.")
        return

    all_results = []

    for extracted_file in tqdm(extracted_files, desc="Evaluating files"):
        file_name = os.path.basename(extracted_file)
        ontology_name = file_name.replace("_test.jsonl", "")
        ground_truth_file = f'data/{dataset}/ground_truth/{file_name}'

        df_gt = pd.read_json(ground_truth_file, lines=True)
        df_pred = pd.read_json(extracted_file, lines=True)
        df_merged = pd.merge(df_gt, df_pred, on='id', suffixes=('_gt', '_pred'))

        # Calculate metrics
        df_merged['precision'] = df_merged.apply(lambda row: get_precision(row['triples_gt'], row['triples_pred']), axis=1)
        df_merged['recall'] = df_merged.apply(lambda row: get_recall(row['triples_gt'], row['triples_pred']), axis=1)
        df_merged['f1'] = df_merged.apply(lambda row: get_f1(row['precision'], row['recall']), axis=1)
        df_merged['sh'] = df_merged.apply(lambda row: get_sh(row['triples_gt'], row['triples_pred']), axis=1)
        df_merged['rh'] = df_merged.apply(lambda row: get_rh(row['triples_gt'], row['triples_pred']), axis=1)
        df_merged['oh'] = df_merged.apply(lambda row: get_oh(row['triples_gt'], row['triples_pred']), axis=1)
        
        ontology_metrics = {
            'ontology': ontology_name,
            'precision': df_merged['precision'].mean(),
            'recall': df_merged['recall'].mean(),
            'f1': df_merged['f1'].mean(),
            'sh': df_merged['sh'].mean(),
            'rh': df_merged['rh'].mean(),
            'oh': df_merged['oh'].mean(),
        }
        all_results.append(ontology_metrics)
        
        # Save detailed results per ontology
        df_merged.to_json(os.path.join(eval_dir, file_name), orient='records', lines=True)

    # Save aggregated results
    df_all_results = pd.DataFrame(all_results)
    agg_path = os.path.join(eval_dir, f'aggregated_results_{dataset}.csv')
    df_all_results.to_csv(agg_path, index=False)

    print(f"Aggregated results for {dataset} saved to {agg_path}")
    print(df_all_results)
    print(f"--- [Step 3] Completed for {model_name} on {dataset} ---")


def main():
    """
    Main function to run the complete benchmark suite.
    """
    start_time = time.time()
    print("======== Starting Full Text2KGBench Suite ========")

    for model in MODELS_TO_RUN:
        for dataset in DATASETS:
            print(f"\n>>>>>> Processing Model: {model}, Dataset: {dataset} <<<<<<")
            step_1_get_llm_responses(model, dataset, test_mode=RUN_IN_TEST_MODE)
            step_2_extract_triples(model, dataset)
            step_3_evaluate_results(model, dataset)
    
    end_time = time.time()
    print("\n======== Text2KGBench Suite Completed ========")
    print(f"Total execution time: {((end_time - start_time) / 60):.2f} minutes")

if __name__ == '__main__':
    main()
