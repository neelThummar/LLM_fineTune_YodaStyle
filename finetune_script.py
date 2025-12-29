import os
import torch
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from lion_pytorch import Lion

from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Environment settings
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
torch.set_float32_matmul_precision('high')

# Prompt templates
template_without_answer = "<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
template_with_answer = template_without_answer + "{answer}<end_of_turn>\n"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load tokenizer and model
model_id = "C:/models/gemma2b"  # Path to your downloaded Gemma 2B model
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True
).to('cuda')


# LoRA adapter config and application
def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
        ],
    )
    return get_peft_model(model, lora_config)

model = apply_lora(model)

print(model.hf_device_map)  # Should show model on 'cuda'
print(model.modules)        # Or inspect layers to confirm 4-bit modules

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params}")
print(f"Total parameters: {total_params}")
print(f"Trainable %: {trainable_params / total_params * 100:.2f}%")

# Chat function for inference
def chat(question, max_new_tokens=32, temperature=0.7, only_answer=False):
    prompt = template_without_answer.format(question=question)
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=100,
            top_p=0.95,
            repetition_penalty=2.0,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_tokens = outputs[0]
    if only_answer:
        output_tokens = output_tokens[input_ids['input_ids'].shape[1]:]

    return tokenizer.decode(output_tokens, skip_special_tokens=True)

# Training loss function
def forward_and_compute_loss(model, tokens, mask, context_length=512):
    tokens = tokens[:, :context_length]
    mask = mask[:, :context_length]
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    mask = mask[:, 1:]
    logits = model(x).logits
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        reduction="none"
    )
    return loss[mask.view(-1)].mean()

# Training loop
CLIP_GRAD_NORM = 1.0
def train(model, dataloader, tokenizer):
    optimizer = Lion(model.parameters(), lr=5e-6)
    model.train()
    step, losses, all_losses = 0, [], []

    for batch in dataloader:
        questions, answers = batch["instruction"], batch["response_style"]
        for i in range(len(questions)):
            question, answer = questions[i], answers[i]
            text = template_with_answer.format(question=question, answer=answer)
            ids = tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            answer_start = text.index(answer)
            mask = ids["offset_mapping"][:, :, 0] >= answer_start
            mask = mask.to(model.device)

            loss = forward_and_compute_loss(model, ids["input_ids"], mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

            losses.append(loss.item())
            all_losses.append(loss.item())

            if step % 100 == 0:
                print("\nSample Eval:", chat("What is the Force?", only_answer=True))
                print(f"Step {step}, Avg Loss: {sum(losses)/len(losses):.4f}")
                losses = []
                checkpoint_path = f"checkpoints/yoda_step_{step}"
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

            step += 1
            if step >= 1000:
                plt.plot(all_losses)
                plt.title("Training Loss Curve")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.show()
                return model

    plt.plot(all_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    return model

# Custom dataset class
def load_style_data(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip().replace("\\n", "\n") for line in f if line.strip()]
    return lines

class StyleDataset(Dataset):
    def __init__(self, responses):
        self.responses = responses

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        return {
            "instruction": "Respond in the style of Yoda.",
            "response_style": self.responses[idx]
        }

# Create dataloaders function
def create_dataloaders(responses, batch_size=8, test_size=0.1, seed=42):
    train_responses, test_responses = train_test_split(
        responses,
        test_size=test_size,
        random_state=seed,
        shuffle=True
    )

    train_dataset = StyleDataset(train_responses)
    test_dataset = StyleDataset(test_responses)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Load Yoda-style dataset
responses = load_style_data("data/yoda.txt")
train_loader, test_loader = create_dataloaders(responses, batch_size=8)

# Start training
print(f"Device: {next(model.parameters()).device}")
model = train(model, train_loader, tokenizer)

# Inference
response = chat("give me a list of all the teams who won the IPL titles up unit now", only_answer=True, max_new_tokens=500)
print(response)

# Merge and save
print("Merging LoRA weights into the base model...")
model = model.merge_and_unload()

save_path = "models/gemma_yoda_style_merged"
print(f"Saving merged model to: {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model and tokenizer saved successfully.")
