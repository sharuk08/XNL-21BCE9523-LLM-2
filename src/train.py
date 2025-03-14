from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

def train_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    dataset = load_dataset("csv", data_files={"train": "data/raw_train.csv", "test": "data/raw_test.csv"})
    
    def tokenize_function(examples):
        inputs = tokenizer(examples['texts'], padding='max_length', truncation=True, max_length=128)
        targets = tokenizer(examples['labels'], padding='max_length', truncation=True, max_length=16)
        inputs['labels'] = targets['input_ids']
        return inputs
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    training_args = TrainingArguments(
        output_dir="models/fine-tuned",
        per_device_train_batch_size=16,
        num_train_epochs=2,
        save_steps=500,
        logging_dir="logs",
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("models/fine-tuned")
    print("Model training complete.")

train_model()