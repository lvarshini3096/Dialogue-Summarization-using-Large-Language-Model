# --- Configuration Constants ---
import torch

MODEL_NAME = 'google/flan-t5-base'
DATASET_NAME = "knkarthick/dialogsum"

# Paths to simulate downloaded checkpoints from the course lab
SFT_CHECKPOINT_PATH = "./flan-dialogue-summary-checkpoint" # Fully Fine-Tuned (Instruct)
PEFT_CHECKPOINT_PATH = "./peft-dialogue-summary-checkpoint-from-s3" # PEFT/LoRA Adapter

# Reward model for RLHF
TOXICITY_MODEL_NAME = "facebook/roberta-hate-speech-dynabench-r4-target"

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
