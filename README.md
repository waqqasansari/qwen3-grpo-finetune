# Qwen3-4B Fine-Tuning with GRPO using Unsloth 🚀

This repository contains a Jupyter Notebook for fine-tuning the [`Qwen3-4B`](https://huggingface.co/Qwen/Qwen1.5-4B) language model using the [Unsloth](https://github.com/unslothai/unsloth) library, with the application of a reward-optimization method called **GRPO** (Guided Reward-based Prompt Optimization). This training setup is optimized for running in Google Colab or locally on 4-bit/16-bit precision hardware.

---

## 🧠 What is GRPO?

**GRPO (Guided Reward-based Prompt Optimization)** is a technique to guide language models toward producing responses in a specific structure or format by rewarding them based on how closely their output matches a desired template. In this notebook:

- A **custom reasoning format** is enforced using a templated prompt style.
- A **reward function** scores model outputs based on how closely they match:
  - the **expected output structure** (e.g., markers like `<think>`, answer tags).
  - the **correctness** of the answer (e.g., numerical similarity, regex extraction).
- This reward function is used to fine-tune the model via Reinforcement Learning-like objectives.

---

## 📘 Notebook Workflow Summary

The notebook goes through the following stages:

### 1. 🔧 Environment Setup
- Installs required libraries: `unsloth`, `vllm`, `peft`, `bitsandbytes`, `datasets`, etc.
- Handles dependency setup differently for **Colab vs Local environments**.

### 2. 🧠 Model Loading
- Loads the **Qwen3-4B-Base** model using Unsloth's `FastLanguageModel`.
- Applies **LoRA adapters** for parameter-efficient fine-tuning.
- Enables 4-bit/16-bit precision for efficient use of GPU memory.

### 3. ✍️ GRPO Chat Template Definition
- Defines a structured prompt format (customizable) to guide reasoning.
- Sets a tokenizer chat template so the model aligns with the structure.

### 4. 🏁 Pre-Fine-Tuning
- Uses a subset of NVIDIA’s [Open Math Reasoning dataset](https://huggingface.co/datasets/nvidia/OpenMathReasoning) to "prime" the model on reasoning-style prompts.
- Truncates long sequences, tokenizes inputs, and formats for training.

### 5. 🎯 GRPO Data Preparation and Reward Function
- Loads datasets like `Open-R1` and optionally `GSM8K`.
- Defines several **reward heuristics**:
  - Format conformity (using symbols and regex).
  - Numerical correctness (e.g., exact match, ratio similarity).
  - Extraction logic for structured answers.

### 6. 🔁 GRPO Fine-Tuning
- Configures and launches training using a **custom GRPO trainer**.
- Trains the model to produce structured, accurate outputs via reward-based learning.

### 7. 🔎 Inference & Comparison
- Tests the model **before and after** GRPO fine-tuning.
- Compares outputs to validate reward-driven improvement.

### 8. 💾 Export & Quantization
- Saves the fine-tuned model in formats compatible with `vLLM`, `GGUF`, and `llama.cpp`.
- Supports quantization for deployment (e.g., `q8_0`, `q4_k_m`).

---

## 🚀 Quick Start

1. Clone the repo and install Python dependencies.
2. Open the notebook: `Qwen3_(4B)_GRPO_fine_tuning.ipynb`
3. Run cells in sequence. Adjust training settings (LoRA rank, data paths, etc.) as needed.
4. Evaluate the model using the provided inference blocks.

---

## 🛠️ Requirements

- Python ≥ 3.10
- CUDA-compatible GPU with 4GB+ VRAM (recommended)
- Hugging Face Token (optional for model downloads)

---

## 📦 Key Technologies

- [Unsloth](https://github.com/unslothai/unsloth)
- [LoRA (PEFT)](https://github.com/huggingface/peft)
- [vLLM](https://github.com/vllm-project/vllm)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)

---

---

## 🙌 Acknowledgements
- Unsloth [Blog](https://unsloth.ai/blog/r1-reasoning)
- Inspired by work from [Unsloth](https://github.com/unslothai/unsloth)
- Model by [Qwen team](https://huggingface.co/Qwen)
- Training data from NVIDIA and Open-R1 datasets
