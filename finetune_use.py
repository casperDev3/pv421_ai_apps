from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="./ivan_potter_model",
    tokenizer="./ivan_potter_model",
    device=0,  # Use GPU if available
)

prompts = [
    "What was the name of the boy wizard who survived??",
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    generated_text = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
    print(f"Generated Text: {generated_text}\n")