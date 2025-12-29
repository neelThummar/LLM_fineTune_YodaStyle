from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

model.save_pretrained("models/gemma_yoda_style_merged")
tokenizer.save_pretrained("models/gemma_yoda_style_merged")
