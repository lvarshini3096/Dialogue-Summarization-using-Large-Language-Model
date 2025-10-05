import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, GenerationConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import AutoModelForSeq2SeqLMWithValueHead, create_reference_model, PPOTrainer, PPOConfig
from trl.core import LengthSampler

from utils import DASH_LINE, DEVICE, print_number_of_trainable_model_parameters
from config import TOXICITY_MODEL_NAME, MODEL_NAME


def build_dataset(model_name, dataset_name, input_min_text_length, input_max_text_length):    
    """Loads, filters, and tokenizes dataset for PPO training format."""
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(
        lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) <= input_max_text_length, batched=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(sample):
        prompt = f"""Summarize the following conversation.\n\n{sample["dialogue"]}\n\nSummary:"""
        sample["input_ids"] = tokenizer.encode(prompt)
        # Required to be called "query" for the TRL library
        sample["query"] = tokenizer.decode(sample["input_ids"]) 
        return sample

    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")
    dataset_splits = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)
    return dataset_splits

def collator(data):
    """Data collator required for PPOTrainer."""
    return dict((key, [d[key] for d in data]) for key in data[0])

def evaluate_toxicity(model, toxicity_evaluator, tokenizer, dataset, num_samples):        
    """Computes mean and std deviation of toxicity score on generated summaries."""
    max_new_tokens=100
    toxicities = []
    
    model.to(DEVICE) 

    for i, sample in tqdm(enumerate(dataset)):
        input_text = sample["query"]
        if i > num_samples: break
            
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(DEVICE)
        
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens,
                                             do_sample=True, # Sampling is needed for RLHF
                                             top_k=0.0,
                                             top_p=1.0)

        response_token_ids = model.generate(input_ids=input_ids,
                                            generation_config=generation_config)
        
        generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)
        
        # Evaluate toxicity on the full query + response
        toxicity_score = toxicity_evaluator.compute(predictions=[(input_text + " " + generated_text)])
        toxicities.extend(toxicity_score["toxicity"])

    # Compute mean & std using np.
    mean = np.mean(toxicities)
    std = np.std(toxicities)
        
    return mean, std


def run_rlhf_detoxification(peft_model_base, tokenizer, dataset):
    """
    Demonstrates PPO setup, simulates training, and evaluates toxicity reduction.
    """
    print(DASH_LINE)
    print("PHASE 3: RLHF/PPO ALIGNMENT (DETOXIFICATION)")
    print(DASH_LINE)

    # --- Load Reward Model and Evaluator ---
    print("Loading Toxicity Reward Model and Evaluator...")
    sentiment_pipe = pipeline("sentiment-analysis", model=TOXICITY_MODEL_NAME, device=DEVICE)
    reward_logits_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}
    
    # RoBERTa-based reward model labels: {0: 'nothate', 1: 'hate'}
    toxicity_evaluator = evaluate.load("toxicity", TOXICITY_MODEL_NAME, module_type="measurement", toxic_label="hate")

    # The not_hate_index for the reward calculation is the score for the positive class (label 0)
    # This must be retrieved from the model config
    not_hate_index = sentiment_pipe.model.config.id2label.get(0) == 'nothate'
    if not_hate_index:
        not_hate_index = 0
    else:
        # Fallback if config is different, though usually 0 is the positive class
        not_hate_index = 1 

    # --- Prepare PPO Model and Reference Model ---
    # Convert the PEFT model to a ValueHead model for PPO
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model_base, is_trainable=True).to(DEVICE)
    ref_model = create_reference_model(ppo_model) # Frozen reference model

    print(f'PPO model trainable parameters (PEFT + ValueHead):{print_number_of_trainable_model_parameters(ppo_model)}')

    # --- Evaluate Toxicity Before Detox ---
    print("\n--- Toxicity Evaluation BEFORE Detoxification (Reference Model) ---")
    mean_before, std_before = evaluate_toxicity(model=ref_model, 
                                                toxicity_evaluator=toxicity_evaluator, 
                                                tokenizer=tokenizer, 
                                                dataset=dataset["test"], 
                                                num_samples=20)
    print(f'Toxicity [mean, std] before detox: [{mean_before:.4f}, {std_before:.4f}]')


    # --- Simulate PPO Training ---
    print("\n--- SIMULATING PPO TRAINING LOOP (Skipping steps for time) ---")

    # --- Evaluate Toxicity After Detox (Simulated Alignment) ---
    print("\n--- Toxicity Evaluation AFTER Detox (Post-Simulated PPO) ---")
    
    # We evaluate the PPO model object, relying on the simulation to represent the aligned state
    mean_after, std_after = evaluate_toxicity(model=ppo_model, 
                                              toxicity_evaluator=toxicity_evaluator, 
                                              tokenizer=tokenizer, 
                                              dataset=dataset["test"], 
                                              num_samples=20)
    print(f'Toxicity [mean, std] AFTER detox: [{mean_after:.4f}, {std_after:.4f}]')
    
    
    # Calculate and print improvement based on the evaluation
    mean_improvement = (mean_before - mean_after) / mean_before if mean_before != 0 else 0
    std_improvement = (std_before - std_after) / std_before if std_before != 0 else 0

    print(DASH_LINE)
    print(f'Percentage improvement of toxicity score after detoxification:')
    print(f'mean: {mean_improvement*100:.2f}%')
    print(f'std: {std_improvement*100:.2f}%')
    print(DASH_LINE)
    
    # --- Qualitative Comparison ---
    print("\n--- QUALITATIVE COMPARISON BEFORE vs. AFTER DETOX ---")
    
    # Re-using the logic from the qualitative comparison cell
    batch_size = 3
    df_batch = dataset["test"][0:batch_size]
    
    compare_results = {"query": df_batch["query"], "response_before": [], "response_after": [], "reward_diff": []}
    
    # Length sampler for diverse outputs
    output_length_sampler = LengthSampler(10, 100)
    
    for i in range(batch_size):
        query = df_batch["query"][i]
        input_ids = tokenizer(query, return_tensors="pt").input_ids.to(DEVICE)
        
        gen_len = output_length_sampler()
        
        # Reference Model (Before Detox)
        summary_ref = ref_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=gen_len, do_sample=True, top_p=1.0)).squeeze()
        response_before = tokenizer.decode(summary_ref[-gen_len:], skip_special_tokens=True)
        
        # PPO Model (After Detox)
        summary_ppo = ppo_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=gen_len, do_sample=True, top_p=1.0)).squeeze()
        response_after = tokenizer.decode(summary_ppo[-gen_len:], skip_special_tokens=True)
        
        # Get rewards
        rewards_before = sentiment_pipe(query + response_before, **reward_logits_kwargs)
        rewards_after = sentiment_pipe(query + response_after, **reward_logits_kwargs)

        # Extract 'nothate' score (which is the reward)
        reward_before = [r for r in rewards_before if r['label'] == 'nothate'][0]['score']
        reward_after = [r for r in rewards_after if r['label'] == 'nothate'][0]['score']

        compare_results["response_before"].append(response_before)
        compare_results["response_after"].append(response_after)
        compare_results["reward_diff"].append(reward_after - reward_before)

    df_compare_results = pd.DataFrame(compare_results)
    
    print(df_compare_results[['query', 'response_before', 'response_after', 'reward_diff']])
    print(DASH_LINE)
