## Dialogue Summarization Pipeline: Prompting, PEFT, and RLHF Detoxification

This project presents a **complete Large Language Model (LLM) pipeline** for **Dialogue Summarization**, integrating three major generative AI techniques:

- **Prompt Engineering**
- **Parameter-Efficient Fine-Tuning (PEFT)**
- **Reinforcement Learning from Human Feedback (RLHF) Detoxification**

The pipeline is built using the **FLAN-T5-base** model and the **DialogSum** dataset.

---

## Project Phases and Core Concepts

| **Phase** | **Technique** | **Goal** | **Key Concepts** |
|------------|---------------|-----------|------------------|
| **1. Prompt Engineering** | Zero-shot, One-shot, Few-shot inference | Establish baseline summarization quality using prompt-based inference | Prompt Engineering, Zero/Few-Shot Inference |
| **2. PEFT Fine-Tuning** | LoRA-based fine-tuning on the summarization task | Improve summarization accuracy with reduced VRAM and faster training | PEFT, Catastrophic Forgetting, ROUGE Metrics |
| **3. RLHF Alignment** | PPO with Toxicity Reward Model | Align model outputs to reduce toxicity | RLHF, Reward Modeling, PPO Alignment |

---

## Project Structure

| **File** | **Description** | **Role in Pipeline** |
|-----------|-----------------|----------------------|
| `main.py` | Entry point that initializes components, loads checkpoints, and runs all phases sequentially | Orchestration |
| `config.py` | Central configuration for constants like model names, dataset IDs, and checkpoint paths | Configuration |
| `utils.py` | Helper functions such as calculating trainable parameters and formatting few-shot prompts | Utilities |
| `phase1_prompting.py` | Core logic for Zero-shot and Few-shot summarization experiments | **Phase 1: Prompt Engineering** |
| `phase2_peft.py` | LoRA fine-tuning implementation and ROUGE-based evaluation | **Phase 2: PEFT Evaluation** |
| `phase3_rlhf.py` | PPO setup, toxicity evaluation, and RLHF loop simulation | **Phase 3: RLHF Alignment** |
| `requirements.txt` | List of dependencies with fixed versions (`peft`, `trl`, etc.) | Dependencies |

---

## Setup and Installation

### **Prerequisites**
- **Python:** 3.9 or later  
- **CUDA:** Recommended for GPU acceleration during PEFT and PPO stages  

---

### Step 1: Clone the Repository**
```bash
git clone https://github.com/your-username/llm-dialogue-pipeline.git
cd llm-dialogue-pipeline
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Acquire Checkpoints

The demo requires model checkpoints to simulate pre-trained and fine-tuned states.
If real checkpoints are unavailable, create placeholder directories:

```bash
mkdir flan-dialogue-summary-checkpoint
mkdir peft-dialogue-summary-checkpoint-from-s3
```

### Usage

To run the entire pipeline—from prompting to PEFT to RLHF alignment—execute:

```bash
python main.py
```
