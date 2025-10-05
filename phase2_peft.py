import numpy as np
import pandas as pd
from transformers import GenerationConfig
from utils import DASH_LINE, DEVICE
import evaluate


def run_peft_evaluation(original_model, instruct_model, peft_model, tokenizer, dataset):
    """
    Loads pre-trained checkpoints and evaluates ROUGE scores for Original, Instruct, and PEFT models.
    NOTE: ROUGE scores are simulated from the final course results due to checkpoint size.
    """
    print(DASH_LINE)
    print("PHASE 2: PEFT FINE-TUNING EVALUATION")
    print(DASH_LINE)

    # --- 2.1 Qualitative Evaluation (Human) ---
    index = 200
    dialogue = dataset['test'][index]['dialogue']
    human_baseline_summary = dataset['test'][index]['summary']
    prompt = f"""Summarize the following conversation.\n\n{dialogue}\n\nSummary:"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # Use max_new_tokens=200, num_beams=1 for simple generation
    gen_config = GenerationConfig(max_new_tokens=200, num_beams=1)

    # Generate outputs
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=gen_config)
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=gen_config)
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=gen_config)
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)


    print("--- QUALITATIVE COMPARISON (Index 200) ---")
    print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
    print(DASH_LINE)
    print(f'ORIGINAL MODEL (Zero-Shot):\n{original_model_text_output}')
    print(DASH_LINE)
    print(f'INSTRUCT MODEL (Full Fine-Tune):\n{instruct_model_text_output}')
    print(DASH_LINE)
    print(f'PEFT MODEL (LoRA Fine-Tune):\n{peft_model_text_output}')
    print(DASH_LINE)


    # --- 2.2 Quantitative Evaluation (Simulated ROUGE) ---
    print("--- QUANTITATIVE COMPARISON (ROUGE Scores - Simulated Full Dataset) ---")
    
    # Simulated ROUGE results
    original_model_results = {'rouge1': 0.2334, 'rouge2': 0.0760, 'rougeL': 0.2015, 'rougeLsum': 0.2015}
    instruct_model_results = {'rouge1': 0.4216, 'rouge2': 0.1804, 'rougeL': 0.3384, 'rougeLsum': 0.3384}
    peft_model_results = {'rouge1': 0.4081, 'rouge2': 0.1633, 'rougeL': 0.3251, 'rougeLsum': 0.3249}

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('INSTRUCT MODEL (Full FT):')
    print(instruct_model_results)
    print('PEFT MODEL (LoRA FT):')
    print(peft_model_results)
    print(DASH_LINE)

    improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
    print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL:")
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f'{key}: {value*100:.2f}%')
    print(DASH_LINE)

    improvement = (np.array(list(peft_model_results.values())) - np.array(list(instruct_model_results.values())))
    print("Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL:")
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f'{key}: {value*100:.2f}%')
    print("\nObservation: The small decrease in ROUGE metrics is usually outweighed by the massive reduction in training time and memory.")
    print(DASH_LINE)
