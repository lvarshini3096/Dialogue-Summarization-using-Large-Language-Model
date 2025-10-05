import torch
from config import DEVICE

# Define global constants used across modules
DASH_LINE = '-' * 100

def print_number_of_trainable_model_parameters(model):
    """Prints the number of trainable parameters and the percentage of all parameters."""
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    
    if all_model_params == 0:
        return "Model has no parameters."

    return (
        f"\ntrainable model parameters: {trainable_model_params}"
        f"\nall model parameters: {all_model_params}"
        f"\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    )

def make_prompt(dataset, example_indices_full, example_index_to_summarize):
    """
    Creates a few-shot prompt string from the dataset examples using the
    FLAN-T5-specific prompt template from the course.
    """
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        # FLAN-T5-specific prompt template 
        # Note: Stop sequence is the start of the next dialogue section
        prompt += f"""Dialogue:\n{dialogue}\nWhat was going on?\n{summary}\n\n"""
    
    dialogue = dataset['test'][example_index_to_summarize]['dialogue']
    prompt += f"""Dialogue:\n{dialogue}\nWhat was going on?\n"""
        
    return prompt
