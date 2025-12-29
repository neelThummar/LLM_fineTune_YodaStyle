# Yoda-Style Language Generation with LoRA-Fine-Tuned Gemma-2B
This project implements parameter-efficient fine-tuning of a large language model to generate responses in the distinct syntactic style of Yoda. A Gemma-2B base model is loaded in 4-bit NF4 precision and fine-tuned using LoRA, enabling efficient training by updating only a small fraction of the model’s parameters.
The project also includes an interactive web-based chat interface that allows users to converse with the model in real time using Gradio.

Fine-tuning large language models is computationally expensive. This project demonstrates how LoRA and quantization enable efficient adaptation of a 1.6B-parameter model using limited resources, while still producing expressive, high-quality stylistic outputs.
It also showcases end-to-end ownership: data handling, model training, merging, and deployment.

## Overview
- Task: Style-conditioned text generation
- Input: Natural language prompts
- Output: Yoda-style responses
- Base Model: Gemma-2B (~1.61B parameters)
- Fine-Tuning Method: LoRA (Parameter-Efficient Fine-Tuning)
- Quantization: 4-bit NF4 (BitsAndBytes)
- Frameworks: PyTorch, Hugging Face Transformers, PEFT
- Deployment: Interactive Gradio web interface

## Model Architecture
- Base Model: Gemma-2B causal language model
- Precision: 4-bit NF4 quantization with FP16 compute
- LoRA Configuration:
    Rank (r): 8
    Alpha: 32
    Dropout: 0.05
    Target Modules:
        Attention projections (q_proj, k_proj, v_proj, o_proj)
        MLP projections (gate_proj, up_proj, down_proj)
- Trainable Parameters: 10.38M (~0.64% of total parameters)
- Final Model: LoRA adapters merged into base model for inference

## Training Configuration
- Optimizer: Lion
- Learning Rate: 5e-6
- Batch size: 8
- Context Length: 512 tokens
- Training Steps: 1,000
- Gradient Clipping: 1.0
- Checkpointing: Periodic weight saving during training
- Loss Function: Cross-entropy loss over target tokens

## Dataset
- Format: Plain-text Yoda-style responses
- Structure: One response per sample
- Ingestion: Custom UTF-8 safe PyTorch data pipeline
- Purpose: Learn stylistic transformation rather than factual QA

## Text Generation
- Generates stylistically consistent Yoda-like responses
- Controlled sampling using:
    Temperature
    Top-k sampling
    Top-p (nucleus) sampling
    Repition penalties
- Produces fluent, grammatically inverted outputs characteristic of Yoda’s speech

## Interactive Web Interface
This project includes an interactive chat application built with Gradio, allowing users to converse with the fine-tuned model in real time. The interface loads the merged Gemma-2B model and enables conversational interaction directly in the browser.

Features
- Live Yoda-style responses
- Adjustable generation parameters
- Runs locally using the merged inference-ready model

## How to Run
1. Install dependencies from `requirements.txt`
2. Run the fine-tuning script - finetune_script.py
3. Launch the chat interface from the link after runing yoda_webchat.py 

 ## Credits
- Base model: Gemma
- Fine-tuning techniques: LoRA (PEFT)
- Deployment: Gradio




