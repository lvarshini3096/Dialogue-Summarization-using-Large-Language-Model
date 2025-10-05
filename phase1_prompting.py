from utils import DASH_LINE, DEVICE, make_prompt
from transformers import GenerationConfig


def run_prompt_engineering_baseline(model, tokenizer, dataset, example_indices):
    """
    Demonstrates Zero-Shot and Few-Shot inference for dialogue summarization
    and the effect of prompt engineering.
    """
    print(DASH_LINE)
    print("PHASE 1: PROMPT ENGINEERING BASELINE")
    print(DASH_LINE)
    print("--- 1.1 Zero-Shot Inference ---")

    for i, index in enumerate(example_indices):
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        # Zero-Shot Prompt template
        zero_shot_prompt = f"""Summarize the following conversation.\n\n{dialogue}\n\nSummary:"""
        
        inputs = tokenizer(zero_shot_prompt, return_tensors='pt').to(DEVICE)
        output = tokenizer.decode(
            model.generate(
                inputs["input_ids"], 
                max_new_tokens=50,
            )[0], 
            skip_special_tokens=True
        )
        
        print(f'Example {i + 1}')
        print(DASH_LINE)
        print(f'BASELINE HUMAN SUMMARY:\n{summary}')
        print(DASH_LINE)
        print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')

    # --- 1.2 Few-Shot Inference Demo ---
    print(DASH_LINE)
    print("--- 1.2 Few-Shot Inference Demo ---")
    
    example_indices_full = [40, 80, 120]
    example_index_to_summarize = 200    # Target dialogue
    
    few_shot_prompt = make_prompt(dataset, example_indices_full, example_index_to_summarize)
    target_summary = dataset['test'][example_index_to_summarize]['summary']

    # Generate output using the combined few-shot prompt
    inputs = tokenizer(few_shot_prompt, return_tensors='pt').to(DEVICE)
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    
    print(f'INPUT PROMPT (showing 3 examples + target dialogue end):\n{few_shot_prompt[-800:]}...')
    print(DASH_LINE)
    print(f'BASELINE HUMAN SUMMARY:\n{target_summary}')
    print(DASH_LINE)
    print(f'MODEL GENERATION - FEW SHOT:\n{output}')
    print(DASH_LINE)
