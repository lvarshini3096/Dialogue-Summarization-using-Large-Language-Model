import os
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

# Import configuration and utilities
from config import MODEL_NAME, SFT_CHECKPOINT_PATH, PEFT_CHECKPOINT_PATH, DATASET_NAME, DEVICE
from utils import DASH_LINE
from phase1_prompting import run_prompt_engineering_baseline
from phase2_peft import run_peft_evaluation
from phase3_rlhf import build_dataset, run_rlhf_detoxification


def main():
    """Initializes models and runs the three project phases."""
    print(DASH_LINE)
    print("🤖 Generative AI Dialogue Summarization & Alignment Pipeline")
    print(DASH_LINE)
    
    # 1. Initialize Base Model and Tokenizer
    print(f"Loading Base Model: {MODEL_NAME}...")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Load Datasets
    print(f"\nLoading Datasets: {DATASET_NAME}...")
    dataset_eval = load_dataset(DATASET_NAME, split='test')
    dataset_ppo = build_dataset(MODEL_NAME, DATASET_NAME, 200, 1000)

    # PHASE 1: PROMPT ENGINEERING
    run_prompt_engineering_baseline(original_model, tokenizer, dataset_eval, [40, 200])

    # --- Load Checkpoints for Phase 2 & 3 ---
    print("\n" + DASH_LINE)
    print("Loading Fine-Tuned Checkpoints (Simulated from Course Data)...")
    
    # SFT Checkpoint (Full Fine-Tune)
    instruct_model = None
    if os.path.isdir(SFT_CHECKPOINT_PATH):
        print(f"Loading Instruct Model (Full FT) from {SFT_CHECKPOINT_PATH}")
        instruct_model = AutoModelForSeq2SeqLM.from_pretrained(SFT_CHECKPOINT_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    else:
        print(f"WARNING: Instruct Checkpoint not found at {SFT_CHECKPOINT_PATH}. Using Base Model for comparison.")
        instruct_model = original_model 

    # PEFT Checkpoint (LoRA)
    peft_model = None
    if os.path.isdir(PEFT_CHECKPOINT_PATH):
        print(f"Loading PEFT Adapter from {PEFT_CHECKPOINT_PATH}")
        peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
        peft_model = PeftModel.from_pretrained(peft_model_base, PEFT_CHECKPOINT_PATH, torch_dtype=torch.bfloat16, is_trainable=False).to(DEVICE)
    else:
        print(f"WARNING: PEFT Checkpoint not found at {PEFT_CHECKPOINT_PATH}. Using Base Model for comparison.")
        peft_model = original_model

    # PHASE 2: PEFT EVALUATION
    run_peft_evaluation(original_model, instruct_model, peft_model, tokenizer, dataset_eval)

    # PHASE 3: RLHF DETOXIFICATION
    run_rlhf_detoxification(peft_model, tokenizer, dataset_ppo)


if __name__ == "__main__":
    if DEVICE == "cpu":
        print("WARNING: GPU not detected. Running on CPU will be extremely slow for fine-tuning/RLHF steps.")
    
    # Set logging to pandas to avoid truncation in printouts
    pd.set_option('display.max_colwidth', 500)
    
    main()
