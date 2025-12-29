import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "models/gemma_yoda_style_merged",  # Local path to merged model
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "models/gemma_yoda_style_merged"
)


# Gradio chat function
def chat_yoda_style(message, history=None):
    inputs = tokenizer(message, return_tensors="pt").to(base_model.device)
    outputs = base_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(message) :].strip()


# Gradio interface
gr.ChatInterface(chat_yoda_style, title="Chat in Yoda Style 🟢", theme="default").launch()
