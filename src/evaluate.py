from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("models/final_model")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Load test dataset
test_ds = load_dataset("csv", data_files={"test": "data/test.csv"})["test"]

# Prediction function
def predict(text):
    input_text = "sentiment: " + text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(inputs["input_ids"], max_length=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate accuracy
predictions = [predict(sample) for sample in test_ds["input"]]
accuracy = sum(1 for pred, true in zip(predictions, test_ds["output"]) if pred == true) / len(predictions)
print(f"Accuracy: {accuracy}")