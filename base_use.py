from transformers import pipeline

generator_base = pipeline(
    "text-generation",
    model="gpt2",
    tokenizer="gpt2",
    device=0,  # Use GPU if available
)

prompts = [
    "who wizard defeated Lord Voldemort?",
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    generated_text = generator_base(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
    print(f"Generated Text: {generated_text}\n")