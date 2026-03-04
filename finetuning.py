import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForLanguageModeling, TrainingArguments
)

# Load the dataset
dataset = load_dataset("json", data_files="potter_facts.jsonl", split="train")

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set training arguments
training_args = {
    "output_dir": "./ivan_potter_model",
    # "overwrite_output_dir": True,
    "num_train_epochs": 35,
    "per_device_train_batch_size": 2,
    "learning_rate": 5e-5,
    "save_steps": 10,
    "logging_steps": 1,
    "fp16": torch.cuda.is_available()
}
training_args = TrainingArguments(**training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./ivan_potter_model")
tokenizer.save_pretrained("./ivan_potter_model")
print("Done fine-tuning and saved the model to ./ivan_potter_model")