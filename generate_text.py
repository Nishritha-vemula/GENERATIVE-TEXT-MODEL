from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Cache for loaded models
loaded_models = {}

def generate_text(prompt, model_name='gpt2', max_length=250, temperature=0.7, top_k=50, top_p=0.95):
    # Load model & tokenizer if not already loaded
    if model_name not in loaded_models:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()
        loaded_models[model_name] = (tokenizer, model)
    else:
        tokenizer, model = loaded_models[model_name]

    # Encode prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,  # 👈 Add this line
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
