from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Load dataset
dataset = load_dataset("csv", data_files={"train": "C:/Users/shaik/OneDrive/Desktop/XNL/data/raw_train.csv", "test": "C:/Users/shaik/OneDrive/Desktop/XNL/data/raw_test.csv"})

# Preprocessing function
def preprocess_function(examples):
    # Replace None values with empty strings
    inputs = ["sentiment: " + (text if text is not None else "") for text in examples["texts"]]
    targets = ["positive" if label == 4 else "negative" for label in examples["label"]]
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=16, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="models/fine-tuned",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    fp16=True,  # Mixed precision training
    save_steps=500,
    logging_dir="experiments/tensorboard",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("models/fine-tuned")
tokenizer.save_pretrained("models/fine-tuned")