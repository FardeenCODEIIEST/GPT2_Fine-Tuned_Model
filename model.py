import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_text(model, tokenizer, sequence, max_length, num_return_sequences=5):
    model = model.to(device)
    inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs,
        do_sample=True,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        num_beams=num_return_sequences,
        top_k=50,
        top_p=0.85,
        pad_token_id=tokenizer.eos_token_id
    )
    return [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]


def arbiter_generate(model, tokenizer, input_prompt, max_length):
    r = random.randint(0, 4)
    text = generate_text(model, tokenizer, input_prompt, max_length)
    response_text = text[r]
    return response_text
